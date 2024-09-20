import express from 'express';
import { downloadMedia } from '../utils/mediaDownloader';
import fs from 'fs';
import {
  transcribeAudio,
  transcribeNonStaticAudio,
} from '../services/transcribeAudio';
import { improveAccessibilityController } from '../controllers/accesibilityController';
import { handleAIResponse } from '../controllers/chatBotController';
import { kbbiController } from '../controllers/kbbiController';

const router = express.Router();


router.post('/', handleAIResponse);


router.post<{}, {}>('/transcribe', async (req, res) => {
  let localFilePath: string | null = null;
  try {
    const { mediaUrl } = req.body;

    if (!mediaUrl) {
      return res.status(400).json({ error: 'Media URL is required' });
    }

    localFilePath = await downloadMedia(mediaUrl);
    const transcript = await transcribeAudio(localFilePath);

    res.json({ transcript });
  } catch (error) {
    console.error('Transcription error:', error);
    res.status(500).json({ error: 'An error occurred during transcription' });
  } finally {
    // Clean up the temporary file
    if (localFilePath && fs.existsSync(localFilePath)) {
      fs.unlinkSync(localFilePath);
    }
  }
});

router.post<{}, {}>('/transcribe-audio', async (req, res) => {
  try {
    const bodyData = JSON.parse(req.body);
    const base64Audio = bodyData.audioData;

    const audioBuffer = Buffer.from(base64Audio, 'base64');

    // Use OpenAI API to transcribe the audio
    const transcription = transcribeNonStaticAudio(audioBuffer);

    // Send the transcription text as response
    res.json({ transcription: transcription });
  } catch (error) {
    console.error('Error during transcription:', error);
    res.status(500).send('Error during transcription');
  }
});

router.get<{}, {}>('/kbbi', kbbiController);

router.post<{}, {}>('/improve-accessibility', improveAccessibilityController);

// Define Document class as needed

export default router;
