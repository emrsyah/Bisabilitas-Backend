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
          { type: 'text', text: 'Kamu adalah sebuah Kamus KBBI, saya akan memberika input berupa kata dan kamu akan memberikan definisinya, HANYA BERIKAN DEFINISINYA SAJA, jangan berikan apapun lagi. Apabila memang yang diberikan bukan sebuah kata valid, maka berikan pesan error seperti ini: "Tak dapat menemukan definisi kata {kata}". Jika kata-kata punya lebih dari 1 arti, pisahkan saja dengan tanda titik koma (;)' },
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
