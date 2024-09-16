import axe from 'axe-core';

export interface Violation {
  id: string;
  description: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  help: string;
  helpUrl: string;
  element: ViolationElement[]
}

export interface ViolationElement {
  html: string;
  target: axe.UnlabelledFrameSelector;
  improvement: string;
}
