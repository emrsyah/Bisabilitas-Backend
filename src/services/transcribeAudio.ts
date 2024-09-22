import { OpenAI, toFile } from 'openai';
import fs from 'fs';

export async function transcribeAudio(filePath: string) {
  //   console.log('OPENAI_API_KEY:', process.env.OPENAI_API_KEY);

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  try {
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(filePath),
      model: 'whisper-1',
      response_format: 'verbose_json',
      timestamp_granularities: ['segment', 'word'],
    });

    // The transcription object now directly contains the segments
    // const segments = transcription.segments.map((segment: any) => ({
    //   start: segment.start,
    //   end: segment.end,
    //   text: segment.text,
    //   words: segment.words.map((word: any) => ({
    //     word: word.word,
    //     start: word.start,
    //     end: word.end,
    //   })),
    // }));

    console.log(transcription);

    // return segments;
  } catch (error) {
    console.error('OpenAI API error:', error);
    throw error;
  } finally {
    // Clean up the temporary file
    fs.unlinkSync(filePath);
  }
}

export async function transcribeNonStaticAudio(buf: Buffer) {
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const transcription = await openai.audio.transcriptions.create({
    file: await toFile(buf, 'audio.wav', {
      type: 'audio/wav',
    }),
    model: 'whisper-1',
  });
  return transcription.text;
}
