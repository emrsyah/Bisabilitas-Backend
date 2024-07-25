import express from 'express';
import {
  PuppeteerWebBaseLoader,
  Page,
  Browser,
} from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { pull } from 'langchain/hub';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Document } from 'langchain/document';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import * as cheerio from 'cheerio';
import { PromptTemplate } from '@langchain/core/prompts';
import { DocumentInterface } from '@langchain/core/documents';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

const router = express.Router();

type EmojiResponse = string[];

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
          return bodyElement?.textContent ?? '';
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

  const documents = [new Document({ pageContent: docsC })];
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
  return split;
};

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

const pineconeIndex = pinecone.Index('bisabilitas-1024');

export interface TypedRequestBody<T> extends Express.Request {
  body: T
}

interface AiRequestBody {
  url: string;
  question: string;
  history: Record<string, any>;
}

router.get<{}, EmojiResponse>('/', async (req: TypedRequestBody<AiRequestBody>, res) => {

  const { url, history, question } = req.body;


  const vectorDimension = 1024;  // Adjust this to your model's embedding size
  const zeroVector = new Array(vectorDimension).fill(0);

  const existingVectors = await pineconeIndex.namespace(url).query({
    vector: zeroVector,
    topK: 3,
  });

  console.log('existing vec', existingVectors.matches);
  console.log('vec length', existingVectors.matches.length);

  const documents = await scrapeAndCleanData(url);
  const splitted = await splitDocuments(documents);  

  const vectorStore = await PineconeStore.fromDocuments(
    splitted,
    new MistralAIEmbeddings(),
    {
      pineconeIndex,
      namespace: req.body.url,
    },
  );

  // const vectorStore = await MemoryVectorStore.fromDocuments(
  //   splitted,
  //   new MistralAIEmbeddings(),
  // );

  const retriever = vectorStore.asRetriever();

  const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use four sentences maximum and keep the answer as concise as possible.
Always provide the evidence of the context that you used for your response.

{context}

Question: {question}

Helpful Answer:`;

  const customRagPrompt = PromptTemplate.fromTemplate(template);

  const prompt = await pull<ChatPromptTemplate>('rlm/rag-prompt');

  const llm = new ChatMistralAI({
    model: 'mistral-large-latest',
    temperature: 0,
  });

  const ragChain = await createStuffDocumentsChain({
    llm,
    prompt: customRagPrompt,
    outputParser: new StringOutputParser(),
  });

  const retrievedDocs = await retriever.invoke(
    question,
  );

  console.log('lenght context', retrievedDocs.length);
  console.log('lenght context', retrievedDocs[0]);
  // console.log('context diambil id', retrievedDocs[0].id);
  // console.log('context diambil', retrievedDocs[0].pageContent);
  // console.log('context diambil metadata', retrievedDocs[0].metadata);

  const coba = await ragChain.invoke({
    question: question,
    context: retrievedDocs,
  });

  console.log(coba);

  // const sequence = splitter.pipe(transformer);

  // const newDocuments = await sequence.invoke(docs);

  // console.log(split);
  res.json(['😀', '😳', '🙄']);
});

export default router;
