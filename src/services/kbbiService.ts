import OpenAI from 'openai';

export const getKbbiDefinition = async (word: string) => {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'system',
        content: [
          { type: 'text', text: 'Kamu adalah sebuah Kamus KBBI, saya akan memberika input berupa kata-kata dan kamu akan memberikan definisinya, HANYA BERIKAN DEFINISI NYA SAJA, jangan berikan apapun lagi.' },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'text', text: word },
        ],
      },
    ],
  });
  return response.choices[0].message.content;
};
