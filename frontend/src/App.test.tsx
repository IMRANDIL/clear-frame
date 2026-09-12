import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from './App'

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ status: 'ready', cuda_available: true, gpu: 'Test GPU' }),
  }))
  vi.stubGlobal('URL', {
    createObjectURL: vi.fn(() => 'blob:preview'),
    revokeObjectURL: vi.fn(),
  })
})

describe('ClearFrame workspace', () => {
  it('starts in image mode and switches to video controls', () => {
    render(<App />)
    expect(screen.getByRole('tab', { name: /image/i })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText('Output scale')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: /video/i }))

    expect(screen.getByRole('tab', { name: /video/i })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByText(/Aspect-safe H.264/)).toBeInTheDocument()
  })

  it('rejects unsupported dropped files with a clear message', () => {
    render(<App />)
    const dropZone = screen.getByRole('button', { name: /Drop an image here/i })
    fireEvent.drop(dropZone, { dataTransfer: { files: [new File(['x'], 'bad.txt')] } })
    expect(screen.getByRole('alert')).toHaveTextContent('Choose PNG, JPG, JPEG, MP4, MOV, or WebM.')
  })
})
