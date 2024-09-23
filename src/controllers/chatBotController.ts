import { Request, Response } from 'express';
import { scrapeAndCleanData } from '../utils/webScraper';
import { createRAGChain, getOrCreateVectorStore, initializeAIComponents } from '../services/chatBotService';
import { AiRequestBody, AiResponse } from '../interfaces/ChatBot';
import { aiResponseSchema } from '../schemas/aiChatbotSchemas';

export const handleAIResponse = async (req: Request<{}, AiResponse, AiRequestBody>, res: Response) => {
  try {
    const { url, history, question } = req.body;
    const { llm, embeddings, pineconeIndex, reranker } = initializeAIComponents();

    const vectorStore = await getOrCreateVectorStore(url, pineconeIndex, embeddings);
    console.log(vectorStore);
    const ragChain = createRAGChain(llm, vectorStore, reranker);

    const result = await ragChain.invoke({ question, chat_history: history, url: url });
    const parsedResult = aiResponseSchema.parse(result);

    const updatedHistory = [
      ...history,
      { role: 'user', content: question, evidence: [] },
      { role: 'ai', content: parsedResult.answer, evidence: parsedResult.evidence },
    ];

    res.json({
      answer: { answer: parsedResult.answer, evidence: parsedResult.evidence },
      history: updatedHistory,
    });
  } catch (error) {
    console.error('Error in handleAIResponse:', error);
    res.status(500).json({ error: error });
  }
};

export const handleTestResponse = async (req: Request<{}, {}, AiRequestBody>, res: Response) => {
  const { url } = req.body;
  const docs = await scrapeAndCleanData(url);
  res.json({ docs: docs[0].pageContent });
};