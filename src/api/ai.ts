import express from 'express';
import {
  PuppeteerWebBaseLoader,
  Page,
  Browser,
} from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { Document } from 'langchain/document';
import { JsonOutputParser, StringOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import * as cheerio from 'cheerio';
import { ChatPromptTemplate, MessagesPlaceholder, PromptTemplate } from '@langchain/core/prompts';
import { DocumentInterface } from '@langchain/core/documents';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { z } from 'zod';


const router = express.Router();

const scrapeAndCleanData = async (url: string):  Promise<Document<Record<string, any>>[]> => {
  const loader = new PuppeteerWebBaseLoader(
    url,
    {
      launchOptions: {
        headless: 'new',
      },
      async evaluate(page: Page, browser: Browser) {
        const textContent = await page.evaluate(() => {
          const bodyElement = document.querySelector('body');
          return bodyElement?.innerText ?? '';
        });
        await browser.close();
        return textContent ?? '';
      },
    },
  );

  const docs = await loader.load();
  const pageContent = docs[0].pageContent;

  // console.log(pageContent);

  const $ = cheerio.load(pageContent);

  $('script, style').remove();

  const cleanedText = $('body')
    .html()
    ?.replace(/<style[^>]*>.*<\/style>/gms, '');
  const cleaned$ = cheerio.load(cleanedText!);

  const textContent = cleaned$('body').text();

  const docsC = textContent.replace(/[^\x20-\x7E]+/g, '');

  const documents = [new Document({ pageContent: docsC, metadata: {
    url: url,
  } })];
  return documents;
};

const splitDocuments = async (documents: Document<Record<string, any>>[]) : Promise<DocumentInterface<Record<string, any>>[]>  => {
  const transformer = new HtmlToTextTransformer();
  // const transformer2 = new MozillaReadabilityTransformer();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  // const transformer = new HtmlToTextTransformer();
  const sequence = splitter.pipe(transformer);
  const split = await sequence.invoke(documents);
  // console.log('split', split);
  return split;
};

export interface TypedRequestBody<T> extends Express.Request {
  body: T
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface AiRequestBody {
  url: string;
  question: string;
  history: ChatMessage[];
}

type AiResponse = {
  answer: string;
  history: ChatMessage[];  
};



const getOrCreateVectorStore = async (url: string, pineconeIndex: Index<RecordMetadata>, embeddings: MistralAIEmbeddings | GoogleGenerativeAIEmbeddings): Promise<PineconeStore> => {
  let vectorDimension: number;
  if (process.env.AI_PROVIDER == 'GEMINI') {
    vectorDimension = 768;
  } else {
    vectorDimension = 1024;
  }
  const zeroVector = new Array(vectorDimension).fill(0);

  const existRecords = await pineconeIndex.query({
    vector: zeroVector,
    topK: 1,
    filter: { url: { $eq: url } },
    includeMetadata: true,
  });

  // console.log(existRecords.matches);

  let vectorStore: PineconeStore;

  if (existRecords.matches.length === 0) {
    // console.log('Creating new vector store for URL:', url);
    const documents = await scrapeAndCleanData(url);
    const splitted = await splitDocuments(documents);
    // console.log('Splitted documents:', splitted);

    // Create a new vector store and wait for it to be populated
    vectorStore = await PineconeStore.fromDocuments(splitted, embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
      textKey: 'text',
    });

    // Ensure the newly created vectors are immediately available
    // await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for 1 second to ensure indexing is complete
  } else {
    // console.log('Using existing vector store for URL:', url);
    vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
      textKey: 'text',
    });
  }

  // Verify that vectors are present
  // const testQuery = await vectorStore.similaritySearch('test query', 1);
  // console.log('Test query result:', testQuery);

  return vectorStore;
};

interface AIComponents {
  llm: ChatMistralAI | ChatGoogleGenerativeAI;
  pineconeIndex: any; // Adjust the type according to Pinecone's Index type
  embeddings: MistralAIEmbeddings | GoogleGenerativeAIEmbeddings;
}

const aiResponseSchema = z.object({
  answer: z.string().describe('the main response/answer'),
  evidence: z.array(z.string()).describe('parts of the text from the context used to ground the answer'),
});

const initializeAIComponents = (): AIComponents => {
  const aiProvider = process.env.AI_PROVIDER;

  const pinecone = new Pinecone();
  let llm: ChatMistralAI | ChatGoogleGenerativeAI;
  let pineconeIndex: any; // Adjust the type according to Pinecone's Index type
  let embeddings: MistralAIEmbeddings | GoogleGenerativeAIEmbeddings;

  // console.log(Google)

  if (aiProvider === 'GEMINI') {
    embeddings = new GoogleGenerativeAIEmbeddings();
    llm = new ChatGoogleGenerativeAI({
      model: 'gemini-pro',
      temperature: 0.4,
    });
    pineconeIndex = pinecone.Index('bisabilitas-768');
  } else {
    llm = new ChatMistralAI({
      model: 'mistral-large-latest',
      temperature: 0.4,
    });
    embeddings = new MistralAIEmbeddings();
    pineconeIndex = pinecone.Index('bisabilitas-1024');
  }
  // llm = llm.withStructuredOutput();
  return { llm, pineconeIndex, embeddings };
};


router.get<{}, AiResponse>('/', async (req: TypedRequestBody<AiRequestBody>, res) => {

  const { llm, embeddings, pineconeIndex } = initializeAIComponents();

  // const structuredLLM = llm.withStructuredOutput(aiResponseSchema);


  const { url, history, question } = req.body;

  const vectorStore = await getOrCreateVectorStore(url, pineconeIndex, embeddings);

  const retriever = vectorStore.asRetriever();

  // console.log()

  // Contextualize Question Chain
  const contextualizeQSystemPrompt = `Given a chat history and the latest user question
  which might reference context in the chat history, formulate a standalone question
  which can be understood without the chat history. Do NOT answer the question,
  just reformulate it if needed and otherwise return it as is.`;

  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ['system', contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ['human', '{question}'],
  ]);

  
  const contextualizeQChain = contextualizeQPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());
  

  const structuredOutputParser = new JsonOutputParser<z.infer<typeof aiResponseSchema>>();
  
  // RAG TEMPLATE
  const template = `
  Use the provided context to answer the question below. Dont fabricate the response. 
  Keep your response concise, with a maximum of four sentences. Respond in JSON format which would be further explained at the end of this prompt.
  
  Context: {context}
  
  Question: {question}
  
  Respond ONLY with a JSON object in the format below:
  answer: "string"  // The main answer to the question, derived from the context
  evidence: ["string1", "string2", ...]  // An array of strings containing the context references used for the answer

  JSON Response:
  `;
  

  const customRagPrompt = PromptTemplate.fromTemplate(template);

  console.log('retriever', await retriever.invoke(question));

  const ragChain = RunnableSequence.from([
    RunnablePassthrough.assign({
      context: (input: Record<string, unknown>) => {
        if (input.chat_history && (input.chat_history as any[]).length > 0) {
          return contextualizeQChain.pipe(retriever).pipe(formatDocumentsAsString).invoke(input);
        }
        return retriever.invoke(input.question as string).then(formatDocumentsAsString);
      },
    }),
    customRagPrompt,
    llm,
    structuredOutputParser,
  ]);

  
  const result = await ragChain.invoke({
    question,
    chat_history: history,
  });
  console.log('resss', result);

  const updatedHistory: ChatMessage[] = [
    ...history,
    { role: 'user', content: question },
    { role: 'assistant', content: result.answer },
  ];

  res.json({
    answer: result.answer,
    history: updatedHistory,
  });

});

export default router;
