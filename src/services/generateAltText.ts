import OpenAI from 'openai';

export const generateAltText = async (url: string) => {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const response = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Buatkan alt-text mendetail untuk menjelaskan gambar ini, maksimal 2 kalimat.' },
          {
            type: 'image_url',
            image_url: {
              url: url,
            },
          },
        ],
      },
    ],
  });
  return (response.choices[0].message.content);
//   return (response.choices[0]);
};
