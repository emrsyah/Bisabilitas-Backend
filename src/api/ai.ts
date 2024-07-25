import express from 'express';
import { PuppeteerWebBaseLoader } from '@langchain/community/document_loaders/web/puppeteer';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { pull } from 'langchain/hub';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { HtmlToTextTransformer } from '@langchain/community/document_transformers/html_to_text';

const router = express.Router();

type EmojiResponse = string[];

router.get<{}, EmojiResponse>('/', async (req, res) => {
  const loader = new PuppeteerWebBaseLoader('https://docs.pinecone.io/reference/api/introduction');

  const docs = await loader.load();

  const transformer = new HtmlToTextTransformer();


  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  // const transformer = new HtmlToTextTransformer();
  const sequence  = splitter.pipe(transformer);
  const split = await sequence.invoke(docs);
  console.log('split length', split.length);
  console.log('page content', split[0].pageContent.length);
  console.log('meta 1', split[0].metadata);
  console.log('meta 2', split[2].metadata);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    split,
    new MistralAIEmbeddings(),
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

  const retrievedDocs = await retriever.invoke('how i can doing aunthetication in pinecone?');

  console.log('lenght context', retrievedDocs.length);
  console.log('context diambil id', retrievedDocs[0].id);
  console.log('context diambil', retrievedDocs[0].pageContent);
  console.log('context diambil metadata', retrievedDocs[0].metadata);

  const coba = await ragChain.invoke({
    question: 'how i can doing aunthetication in pinecone?',
    context: retrievedDocs,
  });

  console.log(coba);

  // const sequence = splitter.pipe(transformer);

  // const newDocuments = await sequence.invoke(docs);

  // console.log(split);
  res.json(['😀', '😳', '🙄']);
});

export default router;
