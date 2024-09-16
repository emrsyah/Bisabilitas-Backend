import { CohereRerank } from '@langchain/cohere';
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatMistralAI, MistralAIEmbeddings } from '@langchain/mistralai';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Index, Pinecone, RecordMetadata } from '@pinecone-database/pinecone';
import { scrapeAndCleanData, splitDocuments } from '../utils/webScraper';
import { PineconeStore } from '@langchain/pinecone';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';
import { JsonOutputParser, StringOutputParser } from '@langchain/core/output_parsers';
import { aiResponseSchema } from '../schemas/aiChatbotSchemas';
import { ChatPromptTemplate, MessagesPlaceholder, PromptTemplate } from '@langchain/core/prompts';

type AIProvider = 'GEMINI' | 'MISTRAL' | 'OPENAI';

interface AIComponents {
  llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI;
  pineconeIndex: Index<RecordMetadata>;
  embeddings: MistralAIEmbeddings | GoogleGenerativeAIEmbeddings | OpenAIEmbeddings;
  reranker: CohereRerank;
}

const AI_PROVIDER_CONFIG: Record<AIProvider, {
  embeddings: new () => MistralAIEmbeddings | GoogleGenerativeAIEmbeddings | OpenAIEmbeddings,
  llm: new () => ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI,
  indexName: string,
  vectorDimension: number
}> = {
  GEMINI: {
    embeddings: GoogleGenerativeAIEmbeddings,
    llm: ChatGoogleGenerativeAI,
    indexName: 'bisabilitas-768',
    vectorDimension: 768,
  },
  MISTRAL: {
    embeddings: MistralAIEmbeddings,
    llm: ChatMistralAI,
    indexName: 'bisabilitas-1024',
    vectorDimension: 1024,
  },
  OPENAI: {
    embeddings: OpenAIEmbeddings,
    llm: ChatOpenAI,
    indexName: 'bisabilitas-1536',
    vectorDimension: 1536,
  },
};

const createContextualizeQChain = (llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI) => {
  const contextualizeQSystemPrompt = `Given a chat history and the latest user question
      which might reference context in the chat history, formulate a standalone question
      which can be understood without the chat history. Do NOT answer the question,
      just reformulate it if needed and otherwise return it as is.`;
  
  const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ['system', contextualizeQSystemPrompt],
    new MessagesPlaceholder('chat_history'),
    ['human', '{question}'],
  ]);
  
  return contextualizeQPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());
};
  
const createCustomRagPrompt = () => {
  const template = `
      You are a chrome extension assistant, your role is to help answering people question.
      Use the provided context to answer the question below. Don't fabricate the response. 
      Keep your response concise, with a maximum of six sentences. Respond in JSON format which would be further explained at the end of this prompt. The maximum evidence you provide is 3 evidence
      
      Context: {context}
      
      Question: {question}
      
      Respond ONLY with a JSON object in the format below:
      answer: "string"  // The main answer to the question, derived from the context
      evidence: ["string1", "string2", ...]  // An array of strings containing the context references used for the answer, the maximum word for each evidence should only be 4 words only and it must a copy-paste text from the context.
  
      JSON Response:
    `;
  
  return PromptTemplate.fromTemplate(template);
};
  
const retrieveAndRerank = async (
  input: Record<string, unknown>,
  retriever: any,
  reranker: CohereRerank,
  contextualizeQChain: any,
) => {
  let docs;
  if (input.chat_history && (input.chat_history as any[]).length > 0) {
    const contextualizedQ = await contextualizeQChain.invoke(input);
    docs = await retriever.invoke(contextualizedQ);
  } else {
    docs = await retriever.invoke(input.question as string);
  }
  
  const rerankedDocs = await reranker.rerank(
    docs,
    input.question as string,
    {
      topN: 5,
    },
  );
  
  return rerankedDocs
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .map((item) => docs[item.index]);
};

export const initializeAIComponents = (): AIComponents => {
  const aiProvider = (process.env.AI_PROVIDER as AIProvider) || 'OPENAI';
  const config = AI_PROVIDER_CONFIG[aiProvider];

  if (!config) {
    throw new Error(`Invalid AI provider: ${aiProvider}`);
  }

  const pinecone = new Pinecone();
  const embeddings = new config.embeddings();
  const llm = new config.llm();
  const pineconeIndex = pinecone.Index(config.indexName);

  const reranker = new CohereRerank({
    model: 'rerank-multilingual-v3.0',
    topN: 5,
  });

  return { llm, pineconeIndex, embeddings, reranker };
};

export const getOrCreateVectorStore = async (
  url: string,
  pineconeIndex: Index<RecordMetadata>,
  embeddings: MistralAIEmbeddings | GoogleGenerativeAIEmbeddings | OpenAIEmbeddings,
): Promise<PineconeStore> => {
  const aiProvider = (process.env.AI_PROVIDER as AIProvider) || 'OPENAI';
  const vectorDimension = AI_PROVIDER_CONFIG[aiProvider].vectorDimension;
  const zeroVector = new Array(vectorDimension).fill(0);

  const existRecords = await pineconeIndex.query({
    vector: zeroVector,
    topK: 1,
    filter: { url: { $eq: url } },
    includeMetadata: true,
  });

  if (existRecords.matches.length === 0) {
    const documents = await scrapeAndCleanData(url);
    const splitted = await splitDocuments(documents);
    console.log('Documents:', documents);
    console.log('Splitted documents:', splitted);

    return PineconeStore.fromDocuments(splitted, embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
      textKey: 'text',
    });
  } else {
    return PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
    });
  }
};

export const createRAGChain = (
  llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI, 
  vectorStore: PineconeStore, 
  reranker: CohereRerank,
) => {
  const retriever = vectorStore.asRetriever();
  const contextualizeQChain = createContextualizeQChain(llm);

  return RunnableSequence.from([
    RunnablePassthrough.assign({
      context: async (input: Record<string, unknown>) => {
        const docs = await retrieveAndRerank(input, retriever, reranker, contextualizeQChain);
        return formatDocumentsAsString(docs);
      },
    }),
    createCustomRagPrompt(),
    llm,
    new JsonOutputParser<typeof aiResponseSchema>(),
  ]);
};
