export type MediaKind = 'image' | 'video'
export type LightingMode = 'auto' | 'force' | 'off'
export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'

export interface HealthResponse {
  status: string
  cuda_available: boolean
  gpu: string | null
}

export interface JobResponse {
  id: string
  media_type: MediaKind
  original_name: string
  status: JobStatus
  progress: number
  stage: string
  message: string
  created_at: string
  updated_at: string
  error_code: string | null
  error_message: string | null
  source_url: string
  output_url: string | null
  report_url: string | null
}

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) message = payload.detail
    } catch {
      // Keep the status-based fallback for non-JSON server failures.
    }
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health')
}

export function getJob(jobId: string): Promise<JobResponse> {
  return request<JobResponse>(`/api/jobs/${jobId}`)
}

export async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' })
  if (!response.ok && response.status !== 404) {
    throw new Error(`Could not remove local job (${response.status})`)
  }
}

export function submitJob(
  file: File,
  mediaType: MediaKind,
  options: { scale: number; lighting: LightingMode },
): Promise<JobResponse> {
  const body = new FormData()
  body.append('file', file)
  body.append('media_type', mediaType)
  body.append('scale', String(options.scale))
  body.append('lighting', options.lighting)
  return request<JobResponse>('/api/jobs', { method: 'POST', body })
}
