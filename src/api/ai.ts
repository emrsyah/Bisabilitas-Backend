import express from 'express';
import { PuppeteerWebBaseLoader, Page, Browser } from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { pull } from 'langchain/hub';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Document } from 'langchain/document';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';
import { MozillaReadabilityTransformer } from '@langchain/community/document_transformers/mozilla_readability';
import * as cheerio from 'cheerio';

const router = express.Router();

type EmojiResponse = string[];

router.get<{}, EmojiResponse>('/', async (req, res) => {
  const loader = new PuppeteerWebBaseLoader('https://js.langchain.com/v0.2/docs/integrations/document_loaders/web_loaders/web_cheerio', {
    launchOptions: {
      headless: 'new',
    },
    async evaluate(page: Page, browser: Browser) {
      const textContent = await page.evaluate(() => {
        const bodyElement = document.querySelector('body');
        return bodyElement?.textContent ?? '' ;
      });
      await browser.close();
      return textContent ?? '';
    },
  });

  const docs = await loader.load();
  const pageContent = docs[0].pageContent;
  const $ = cheerio.load(pageContent);  
  $('script, style').remove();
  const cleanedText = $('body').html()?.replace(/<style[^>]*>.*<\/style>/gms, '');
  const cleaned$ = cheerio.load(cleanedText!);

  const textContent = cleaned$('body').text();

  const docsC = textContent.replace(/[^\x20-\x7E]+/g, '');

  const documents =  [new Document( { pageContent: docsC } )];

  console.log('docs', documents);

  const transformer = new HtmlToTextTransformer();
  const transformer2 = new MozillaReadabilityTransformer();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  // const transformer = new HtmlToTextTransformer();
  const sequence  = splitter.pipe(transformer2);
  const split = await sequence.invoke(documents);
  // console.log('split length', split);
  // console.log('split length', split.length);
  // console.log('page content', split[10].pageContent);
  // console.log('meta 1', split[0].metadata);
  // console.log('meta 2', split[2].metadata);

  const vectorStore = await MemoryVectorStore.fromDocuments(
    split,
    new MistralAIEmbeddings({
      // batchSize: 200000,
    }),
  );

  console.log('berhasil lewat vector store');

  const retriever = vectorStore.asRetriever();
  const prompt = await pull<ChatPromptTemplate>('rlm/rag-prompt');
  const llm = new ChatMistralAI({
    model: 'mistral-large-latest',
    temperature: 0,
  });

  const ragChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });

  const retrievedDocs = await retriever.invoke('what project the owner build that won competition, and what is used for?');

  console.log('lenght context', retrievedDocs.length);
  console.log('context diambil id', retrievedDocs[0].id);
  console.log('context diambil', retrievedDocs[0].pageContent);
  console.log('context diambil metadata', retrievedDocs[0].metadata);

  const coba = await ragChain.invoke({
    question: 'what project the owner build that won competition, and what is used for?',
    context: retrievedDocs,
  });

  console.log(coba);

  // const sequence = splitter.pipe(transformer);

  // const newDocuments = await sequence.invoke(docs);

  // console.log(split);
  res.json(['😀', '😳', '🙄']);
});

export default router;
