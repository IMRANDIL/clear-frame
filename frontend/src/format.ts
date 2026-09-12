import type { MediaKind } from './api'

const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg'])
const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'webm'])

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = units[0]
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024
    unit = units[index]
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${unit}`
}

export function detectMediaKind(filename: string): MediaKind | null {
  const extension = filename.split('.').pop()?.toLowerCase() ?? ''
  if (IMAGE_EXTENSIONS.has(extension)) return 'image'
  if (VIDEO_EXTENSIONS.has(extension)) return 'video'
  return null
}

export function outputLabel(filename: string, mediaKind: MediaKind = 'image'): string {
  const dot = filename.lastIndexOf('.')
  const stem = dot > 0 ? filename.slice(0, dot) : filename
  const extension = mediaKind === 'video' ? '.mp4' : dot > 0 ? filename.slice(dot) : ''
  return `${stem}-enhanced${extension}`
}
