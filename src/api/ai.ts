import express from 'express';
import {
  PuppeteerWebBaseLoader,
} from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { Document } from 'langchain/document';
import {
  JsonOutputParser,
  StringOutputParser,
} from '@langchain/core/output_parsers';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import * as cheerio from 'cheerio';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from '@langchain/core/prompts';
import { DocumentInterface } from '@langchain/core/documents';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from '@langchain/google-genai';
import { z } from 'zod';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { CohereRerank } from '@langchain/cohere';
import { downloadMedia } from '../utils/mediaDownloader';
import fs from 'fs';
import {
  transcribeAudio,
  transcribeNonStaticAudio,
} from '../services/transcribeAudio';
import AxePuppeteer from '@axe-core/puppeteer';
// import axe from 'ax'
import puppeteer from 'puppeteer';
import { generateAltText } from '../services/generateAltText';
import { generateAIImprovement } from '../services/generateAIImprovement';
import { improveAccessibilityController } from '../controllers/accesibilityController';

const router = express.Router();

const scrapeAndCleanData = async (
  url: string,
): Promise<Document<Record<string, any>>[]> => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: 'new',
    },
    // gotoOptions: {
    //   waitUntil: 'networkidle0',
    // },
    async evaluate(page, browser) {
      // Set a random user agent
      // await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');

      // Scroll to bottom to trigger lazy loading
      await page.evaluate(() => {
        window.scrollTo(0, document.body.scrollHeight);
      });

      // Wait for 5 seconds to allow dynamic content to load
      // await page.waitForTimeout(5000);

      const content = await page.content();
      await browser.close();
      return content;
    },
  });

  const docs = await loader.load();
  const pageContent = docs[0].pageContent;

  const $ = cheerio.load(pageContent);

  // Remove scripts, styles, and other non-content elements
  $('script, style, nav, footer, header').remove();

  const cleanedText = $('body').html() ?? '';
  const cleaned$ = cheerio.load(cleanedText);

  const textContent = cleaned$('body').text();

  // Remove non-printable characters
  const docsC = textContent.replace(/[^\x20-\x7E]+/g, ' ').trim();

  const docsCNoEmptyLines = docsC.replace(/^\s*[\r\n]/gm, '');

  const documents = [
    new Document({ pageContent: docsCNoEmptyLines, metadata: { url: url } }),
  ];
  return documents;
};

const splitDocuments = async (
  documents: Document<Record<string, any>>[],
): Promise<DocumentInterface<Record<string, any>>[]> => {
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
  body: T;
}

interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
  evidence: string[];
}

interface AiRequestBody {
  url: string;
  question: string;
  history: ChatMessage[];
}

interface Answer {
  answer: string;
  evidence: string[];
}

type AiResponse = {
  answer: Answer;
  history: ChatMessage[];
};

const getOrCreateVectorStore = async (
  url: string,
  pineconeIndex: Index<RecordMetadata>,
  embeddings:
  | MistralAIEmbeddings
  | GoogleGenerativeAIEmbeddings
  | OpenAIEmbeddings,
): Promise<PineconeStore> => {
  let vectorDimension: number;
  if (process.env.AI_PROVIDER == 'GEMINI') {
    vectorDimension = 768;
  } else if (process.env.AI_PROVIDER == 'MISTRAL') {
    vectorDimension = 1024;
  } else {
    vectorDimension = 1536;
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
    console.log('Documents:', documents);
    console.log('Splitted documents:', splitted);

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
      // textKey: 'text',
    });
  }

  // Verify that vectors are present
  // const testQuery = await vectorStore.similaritySearch('test query', 1);
  // console.log('Test query result:', testQuery);

  return vectorStore;
};

interface AIComponents {
  llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI;
  pineconeIndex: any; // Adjust the type according to Pinecone's Index type
  embeddings:
  | MistralAIEmbeddings
  | GoogleGenerativeAIEmbeddings
  | OpenAIEmbeddings;
  reranker: CohereRerank;
}

const aiResponseSchema = z.object({
  answer: z.string().describe('the main response/answer'),
  evidence: z
    .array(z.string())
    .describe('parts of the text from the context used to ground the answer'),
});

const initializeAIComponents = (): AIComponents => {
  const aiProvider = process.env.AI_PROVIDER;

  const pinecone = new Pinecone();
  let llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI;
  let pineconeIndex: any; // Adjust the type according to Pinecone's Index type
  let embeddings:
  | MistralAIEmbeddings
  | GoogleGenerativeAIEmbeddings
  | OpenAIEmbeddings;

  const reranker = new CohereRerank({
    model: 'rerank-multilingual-v3.0',
    topN: 5,
  });

  // console.log(Google)

  if (aiProvider === 'GEMINI') {
    embeddings = new GoogleGenerativeAIEmbeddings();
    llm = new ChatGoogleGenerativeAI({
      model: 'gemini-1.5-flash',
      temperature: 0.4,
    });
    pineconeIndex = pinecone.Index('bisabilitas-768');
  } else if (aiProvider === 'MISTRAL') {
    llm = new ChatMistralAI({
      model: 'mistral-large-latest',
      temperature: 0.4,
    });
    embeddings = new MistralAIEmbeddings();
    pineconeIndex = pinecone.Index('bisabilitas-1024');
  } else {
    llm = new ChatOpenAI({
      model: 'gpt-4o-mini',
      temperature: 0.4,
    });
    embeddings = new OpenAIEmbeddings();
    pineconeIndex = pinecone.Index('bisabilitas-1536');
  }
  // llm = llm.withStructuredOutput();
  return { llm, pineconeIndex, embeddings, reranker };
};

router.post<{}, AiResponse>(
  '/',
  async (req: TypedRequestBody<AiRequestBody>, res) => {
    const { llm, embeddings, pineconeIndex, reranker } =
      initializeAIComponents();

    // const structuredLLM = llm.withStructuredOutput(aiResponseSchema);

    const { url, history, question } = req.body;

    const vectorStore = await getOrCreateVectorStore(
      url,
      pineconeIndex,
      embeddings,
    );

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

    const structuredOutputParser = new JsonOutputParser<
    z.infer<typeof aiResponseSchema>
    >();

    // RAG TEMPLATE
    const template = `
  You are a chrome extension assistant, your role is to help answering people question.
  Use the provided context to answer the question below. Dont fabricate the response. 
  Keep your response concise, with a maximum of six sentences. Respond in JSON format which would be further explained at the end of this prompt. The maximum evidence you provide is 3 evidence
  
  Context: {context}
  
  Question: {question}
  
  Respond ONLY with a JSON object in the format below:
  answer: "string"  // The main answer to the question, derived from the context
  evidence: ["string1", "string2", ...]  // An array of strings containing the context references used for the answer, the maximum word for each evidence should only be 4 words only and it must a copy-paste text from the context.

  JSON Response:
  `;

    const customRagPrompt = PromptTemplate.fromTemplate(template);

    // console.log('retriever', await retriever.invoke(question));

    const ragChain = RunnableSequence.from([
      RunnablePassthrough.assign({
        context: async (input: Record<string, unknown>) => {
          // if (input.chat_history && (input.chat_history as any[]).length > 0) {
          //   return contextualizeQChain.pipe(retriever).pipe(formatDocumentsAsString).invoke(input);
          // }
          // return retriever.invoke(input.question as string).then(formatDocumentsAsString);

          // ================== START ==================
          let docs;
          if (input.chat_history && (input.chat_history as any[]).length > 0) {
            const contextualizedQ = await contextualizeQChain.invoke(input);
            docs = await retriever.invoke(contextualizedQ);
          } else {
            docs = await retriever.invoke(input.question as string);
          }
          console.log('not ranked===========');
          console.log(docs);

          // Apply Cohere Rerank
          const rerankedDocs = await reranker.rerank(
            docs,
            input.question as string,
            {
              topN: 5,
            },
          );

          console.log('ranked===========');
          console.log(rerankedDocs);

          // Sort the reranked documents by relevance score
          const sortedDocs = rerankedDocs
            .sort((a, b) => b.relevanceScore - a.relevanceScore)
            .map((item) => docs[item.index]);

          // console.log(sortedDocs);

          return formatDocumentsAsString(sortedDocs);

          // ================== ENDED ==================
        },
      }),
      customRagPrompt,
      llm,
      structuredOutputParser,
    ]);

    const transformedHistory = history.map((h) => {
      if (h.role == 'ai') {
        return new AIMessage(h.content);
      } else {
        return new HumanMessage(h.content);
      }
    });

    const result = await ragChain.invoke({
      question,
      chat_history: transformedHistory,
    });

    const updatedHistory: ChatMessage[] = [
      ...history,
      { role: 'user', content: question, evidence: [] },
      { role: 'ai', content: result.answer, evidence: result.evidence },
    ];

    res.json({
      answer: {
        answer: result.answer,
        evidence: result.evidence,
      },
      history: updatedHistory,
    });
  },
);

router.post<{}, {}>(
  '/test',
  async (req: TypedRequestBody<AiRequestBody>, res) => {
    const { url } = req.body;
    const docs = await scrapeAndCleanData(url);
    console.log(docs);
    console.log(docs.length);
    res.json({ docs: docs[0].pageContent });
  },
);


router.post<{}, {}>('/transcribe', async (req, res) => {
  let localFilePath: string | null = null;
  try {
    const { mediaUrl } = req.body;

    if (!mediaUrl) {
      return res.status(400).json({ error: 'Media URL is required' });
    }

    localFilePath = await downloadMedia(mediaUrl);
    const transcript = await transcribeAudio(localFilePath);

    res.json({ transcript });
  } catch (error) {
    console.error('Transcription error:', error);
    res.status(500).json({ error: 'An error occurred during transcription' });
  } finally {
    // Clean up the temporary file
    if (localFilePath && fs.existsSync(localFilePath)) {
      fs.unlinkSync(localFilePath);
    }
  }
});

router.post<{}, {}>('/transcribe-audio', async (req, res) => {
  try {
    const bodyData = JSON.parse(req.body);
    const base64Audio = bodyData.audioData;

    const audioBuffer = Buffer.from(base64Audio, 'base64');

    // Use OpenAI API to transcribe the audio
    const transcription = transcribeNonStaticAudio(audioBuffer);

    // Send the transcription text as response
    res.json({ transcription: transcription });
  } catch (error) {
    console.error('Error during transcription:', error);
    res.status(500).send('Error during transcription');
  }
});

router.post<{}, {}>('/improve-accessibility', improveAccessibilityController);

// Define Document class as needed

export default router;
