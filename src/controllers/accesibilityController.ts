import { Request, Response } from 'express';
import { analyzeWebsite } from '../services/websiteAnalyzer';
import { processViolations } from '../services/violationProcessor';
import { cleanHtml } from '../utils/htmlCleaner';
import { Violation, ViolationElement } from '../interfaces/Accesibility';
import axe from 'axe-core';

function transformViolations(axeResults: axe.AxeResults): Violation[] {
  return axeResults.violations.map((v) => {
    const nodesStructured: ViolationElement[] = v.nodes.map((n) => {
      const cleandedNodes: ViolationElement = {
        html: n.html,
        target: n.target, // This is now of type UnlabelledFrameSelector
        improvement: '',
      };
      return cleandedNodes;
    });

    const structured: Violation = {
      id: v.id,
      description: v.description,
      impact: v.impact ?? 'critical',
      help: v.help,
      helpUrl: v.helpUrl,
      element: nodesStructured, // This is now compatible with ViolationElement[]
    };

    return structured;
  });
}

export async function improveAccessibilityController(req: Request, res: Response) {
  try {
    const { url } = req.body;
    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    const { content, axeResults } = await analyzeWebsite(url);
    const cleanedHtml = cleanHtml(content);
    const finalHtml = cleanedHtml
      .replace(/>\s+</g, '><')
      .replace(/<(\w+)><\/\1>/g, '')
      .trim();

    const transformedViolations = transformViolations(axeResults);

    const finalViolations = await processViolations(transformedViolations, url);

    res.json({ finalViolations });
  } catch (error) {
    console.error('Error during HTML processing:', error);
    res.status(500).json({ error: 'Error during HTML processing' });
  }
}
