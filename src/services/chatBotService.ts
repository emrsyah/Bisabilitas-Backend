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
import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { DocumentInterface } from '@langchain/core/documents';

type AIProvider = 'GEMINI' | 'MISTRAL' | 'OPENAI';

interface AIComponents {
  llm: ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI;
  pineconeIndex: Index<RecordMetadata>;
  embeddings: MistralAIEmbeddings | GoogleGenerativeAIEmbeddings | OpenAIEmbeddings;
  reranker: CohereRerank;
}

const AI_PROVIDER_CONFIG: Record<AIProvider, {
  embeddings: new () => MistralAIEmbeddings | GoogleGenerativeAIEmbeddings | OpenAIEmbeddings,
  llm: new (options?: any) => ChatMistralAI | ChatGoogleGenerativeAI | ChatOpenAI, // Allow options for instantiation
  indexName: string,
  vectorDimension: number
}> = {
  GEMINI: {
    embeddings: GoogleGenerativeAIEmbeddings,
    llm: ChatGoogleGenerativeAI, // Store the constructor only
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
  You are an AI assistant for a Chrome extension designed to answer users' questions using context from a website. Your response must be informative, well-structured, and follow these specific guidelines:
  
  1. Use only the provided context to formulate your answer. Do not add any information beyond what is present in the context.
  2. Provide your response in **Markdown format** for better readability, ensuring it is clear, concise (3-6 sentences), and informative.
  3. Include subheadings if the answer addresses multiple aspects.
  4. Use bullet points or numbered lists if applicable for better clarity.
  5. **Cite evidence** from the context in your answer using numbered references like [1], [2], etc., linking the citation directly to the relevant quote.
  6. You must provide **at least 1 direct quote** from the context for each key part of your answer, with a maximum of 10 words per quote.
  7. Respond to the question in the language that being used in the question.
  8. Structure your response in JSON into two parts: the answer and the evidence. The format should be as follows:
  9. First, identify the evidence you want to use from the context, then answer the question using that evidence.
  10. The evidence MUST BE an EXACT COPY PASTE from the context, preserving the original language of the website.
  11. For evidence, use the first 3-4 words of the relevant context as the citation.
  
  
    "answer": "Your markdown-formatted answer with citations  tags in the form [1], [2], etc. (the citation must be at the end of the sentence that used that citation).",
    "evidence": [
      "[1]: Exact quote from context in original language (max 10 words)",
      "[2]: Another exact quote from context in original language (max 10 words)",
      "[3]: Further exact quote from context in original language (max 10 words)"
    ]
  
  
  Ensure your answer directly addresses the user's question while adhering to these rules. The response should be both accurate and clearly linked to the provided context.
  
  Context: {context}
  
  Question: {question}
  
  Respond ONLY with the JSON object format outlined above.`;

  
  return PromptTemplate.fromTemplate(template);
};
  
const retrieveAndRerank = async (
  input: Record<string, unknown>,
  retriever: VectorStoreRetriever<PineconeStore>,
  reranker: CohereRerank,
  contextualizeQChain: any,
) => {
  let docs: DocumentInterface<Record<string, any>>[];
  if (input.chat_history && (input.chat_history as any[]).length > 0) {
    const contextualizedQ = await contextualizeQChain.invoke(input);
    docs = await retriever.invoke(contextualizedQ);
  } else {
    docs = await retriever.invoke(`this is the website url: ${input.url}, and the question is ` + input.question as string);
  }

  /// ini hasilnya kosong saat pertama kali

  console.log(docs);
  
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

  if (aiProvider === 'GEMINI') {
    (llm as ChatGoogleGenerativeAI).maxOutputTokens = 1024;
    // (llm as ChatGoogleGenerativeAI).temperature = 0.6;
    (llm as ChatGoogleGenerativeAI).model = 'gemini-1.5-flash';
  } else if (aiProvider === 'MISTRAL') {
    (llm as ChatMistralAI).maxTokens = 1024;
    // (llm as ChatMistralAI).temperature = 0.6;
  } else if (aiProvider === 'OPENAI') {
    (llm as ChatOpenAI).maxTokens = 1024;
    // (llm as ChatOpenAI).temperature = 0.6;
    (llm as ChatOpenAI).model = 'gpt-4o';
  }

  console.log('llm', llm.temperature);

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
    console.log('No existing records found. Scraping and creating new vector store...');
    const documents = await scrapeAndCleanData(url);
    const splitted = await splitDocuments(documents);
    // console.log('Documents:', documents);
    // console.log('Splitted documents:', splitted);

    const pineconeStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
    });
    await pineconeStore.addDocuments(splitted);
    return pineconeStore;
  } else {
    const pineconeStore = await PineconeStore.fromExistingIndex(embeddings, {
      pineconeIndex,
      filter: { url: { $eq: url } },
    });
    return pineconeStore;
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
        // console.log('used docs', docs);
        return formatDocumentsAsString(docs);
      },
    }),
    createCustomRagPrompt(),
    llm,
    new JsonOutputParser<typeof aiResponseSchema>(),
  ]);
};
