import { Request, Response } from 'express';
import { scrapeAndCleanData } from '../utils/webScraper';
import { createRAGChain, getOrCreateVectorStore, initializeAIComponents } from '../services/chatBotService';
import { AiRequestBody, AiResponse } from '../interfaces/ChatBot';
import { aiResponseSchema } from '../schemas/aiChatbotSchemas';

export const handleAIResponse = async (req: Request<{}, AiResponse, AiRequestBody>, res: Response) => {
  const { url, history, question } = req.body;
  const { llm, embeddings, pineconeIndex, reranker } = initializeAIComponents();

  const vectorStore = await getOrCreateVectorStore(url, pineconeIndex, embeddings);
  const ragChain = createRAGChain(llm, vectorStore, reranker);

  const result = await ragChain.invoke({ question, chat_history: history });

  const parsedResult = aiResponseSchema.parse(result);

  console.log(parsedResult.answer);
  console.log(parsedResult.evidence);

  //   const updatedHistory = [
  //     ...history,
  //     { role: 'user', content: question, evidence: [] },
  //     { role: 'ai', content: result.answer, evidence: result.evidence },
  //   ];

//   res.json({
//     answer: { answer: result.answer, evidence: result.evidence },
//     history: updatedHistory,
//   });
};

export const handleTestResponse = async (req: Request<{}, {}, AiRequestBody>, res: Response) => {
  const { url } = req.body;
  const docs = await scrapeAndCleanData(url);
  res.json({ docs: docs[0].pageContent });
};