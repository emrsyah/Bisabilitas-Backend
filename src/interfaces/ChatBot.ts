export interface ChatMessage {
  role: 'user' | 'ai';
  content: string;
  evidence: string[];
}
  
export interface AiRequestBody {
  url: string;
  question: string;
  history: ChatMessage[];
}
  
export interface Answer {
  answer: string;
  evidence: string[];
}
  
export type AiResponse = {
  answer: Answer;
  history: ChatMessage[];
};