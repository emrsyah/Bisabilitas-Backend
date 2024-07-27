import express from 'express';
import {
  PuppeteerWebBaseLoader,
  Page,
  Browser,
} from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { Document } from 'langchain/document';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import * as cheerio from 'cheerio';
import { ChatPromptTemplate, MessagesPlaceholder, PromptTemplate } from '@langchain/core/prompts';
import { DocumentInterface } from '@langchain/core/documents';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';

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



const getOrCreateVectorStore = async (url: string, pineconeIndex: Index<RecordMetadata>, embeddings: MistralAIEmbeddings): Promise<PineconeStore> => {
  const vectorDimension = 1024;
  const zeroVector = new Array(vectorDimension).fill(0);

  const existRecords = await pineconeIndex.query({
    vector: zeroVector,
    topK: 1,
    filter: { url: { $eq: url } },
    includeMetadata: true,
  });

  if (existRecords.matches.length === 0) {
    // console.log('ga ada vec');
    const documents = await scrapeAndCleanData(url);
    const splitted = await splitDocuments(documents);
    return PineconeStore.fromDocuments(splitted, embeddings, {
      pineconeIndex,
    });
  } else {
    // console.log('ada vec');
    return PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
    });
  }
};
router.get<{}, AiResponse>('/', async (req: TypedRequestBody<AiRequestBody>, res) => {

  const llm = new ChatMistralAI({
    model: 'mistral-large-latest',
    temperature: 0,
  });

  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index('bisabilitas-1024');

  const embeddings = new MistralAIEmbeddings();

  const { url, history, question } = req.body;

  const vectorStore = await getOrCreateVectorStore(url, pineconeIndex, embeddings);

  const retriever = vectorStore.asRetriever();

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

  // RAG TEMPLATE
  const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use four sentences maximum and keep the answer as concise as possible.
Always provide the evidence of the context that you used for your response.

{context}

Question: {question}

Helpful Answer:`;

  const customRagPrompt = PromptTemplate.fromTemplate(template);

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
    new StringOutputParser(),
  ]);

  const result = await ragChain.invoke({
    question,
    chat_history: history,
  });

  const updatedHistory: ChatMessage[] = [
    ...history,
    { role: 'user', content: question },
    { role: 'assistant', content: result },
  ];

  res.json({
    answer: result,
    history: updatedHistory,
  });

});

export default router;
