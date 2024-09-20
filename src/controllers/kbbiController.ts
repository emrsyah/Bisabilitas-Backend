import { Request, Response } from 'express';
import { getKbbiDefinition } from '../services/kbbiService';

export const kbbiController = async (req: Request, res: Response) => {
  try {
    const { word } = req.body;
    if (!word) {
      return res.status(400).json({ error: 'Word is required' });
    }
    const definition = await getKbbiDefinition(word);
    res.json({ 'definition': definition });
  } catch (err) {
    console.error(err);
  }
};
