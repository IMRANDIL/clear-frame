import { describe, expect, it } from 'vitest'

import { detectMediaKind, formatBytes, outputLabel } from './format'

describe('media formatting', () => {
  it('detects supported file kinds without case sensitivity', () => {
    expect(detectMediaKind('night.PNG')).toBe('image')
    expect(detectMediaKind('clip.WebM')).toBe('video')
    expect(detectMediaKind('notes.txt')).toBeNull()
  })

  it('formats sizes and enhanced filenames', () => {
    expect(formatBytes(1536)).toBe('1.50 KB')
    expect(outputLabel('portrait.jpg')).toBe('portrait-enhanced.jpg')
    expect(outputLabel('clip.webm', 'video')).toBe('clip-enhanced.mp4')
  })
})
