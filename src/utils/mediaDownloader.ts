import axios from 'axios';
import fs from 'fs';
import path from 'path';
import ffmpeg from 'fluent-ffmpeg';
import stream from 'stream';

function extractAudioFromVideo(
  videoStream: stream.Readable,
  outputPath: string,
): Promise<string> {
  console.log('steam===================', videoStream);
  return new Promise((resolve, reject) => {
    ffmpeg(videoStream)
      .noVideo()
      .audioCodec('libmp3lame')
      .audioChannels(1)
      .audioFrequency(16000)
      .format('mp3')
      .on('end', () => resolve(outputPath))
      .on('error', (err) => reject(err))
      .save(outputPath);
  });
}

export async function downloadMedia(url: string): Promise<string> {
  const response = await axios({
    method: 'GET',
    url: url,
    responseType: 'stream',
  });

  const contentType = response.headers['content-type'];
  const isVideo = contentType.includes('video');
  const extension = isVideo ? '.mp4' : '.mp3';
  const fileName = `media_${Date.now()}${extension}`;
  const filePath = path.join(__dirname, '..', '..', 'temp', fileName);
//   console.log('hlo bossssssssssssssssssssssssssssssss');
  if (isVideo) {
    const res = extractAudioFromVideo(response.data, filePath);
    console.log('res11111111111111', res);
    return res;
  } else {
    const writer = fs.createWriteStream(filePath);
    response.data.pipe(writer);
    console.log('res222222222222', writer);
    return new Promise((resolve, reject) => {
      writer.on('finish', () => resolve(filePath));
      writer.on('error', reject);
    });
  }
}
