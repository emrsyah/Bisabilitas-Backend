import OpenAI from 'openai';

export const generateAIImprovement = async (element: string, prompt: string) => {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: prompt + ' only return the element after improved in plain text, no other things!' },
          {
            type: 'text',
            text: element,
          },
        ],
      },
    ],
  });
  return (response.choices[0].message.content);
//   return (response.choices[0]);
};
