import { z } from 'zod';

export const aiResponseSchema = z.object({
  answer: z.string().describe('the main response/answer'),
  evidence: z.array(z.string()).describe('parts of the text from the context used to ground the answer'),
});