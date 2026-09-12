import { useEffect, useRef, useState } from 'react'

import { deleteJob, getHealth, getJob, submitJob } from './api'
import type { HealthResponse, JobResponse, LightingMode, MediaKind } from './api'
import { detectMediaKind, formatBytes, outputLabel } from './format'

const TERMINAL_STATES = new Set(['completed', 'failed'])

function Mark() {
  return (
    <svg aria-hidden="true" className="mark" viewBox="0 0 38 38">
      <path d="M5 5h11v4H9v7H5V5Zm17 0h11v11h-4V9h-7V5ZM5 22h4v7h7v4H5V22Zm24 0h4v11H22v-4h7v-7Z" />
      <path d="m13 20 4 4 9-11 3 3-12 14-7-7 3-3Z" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 40 40">
      <path d="M20 5v20m0-20-8 8m8-8 8 8M7 25v8h26v-8" />
    </svg>
  )
}

function SparkIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M12 2c.5 5.6 2.9 8 8.5 8.5C14.9 11 12.5 13.4 12 19c-.5-5.6-2.9-8-8.5-8.5C9.1 10 11.5 7.6 12 2Z" />
      <path d="M19 16c.2 2.2 1.2 3.2 3.5 3.5-2.3.2-3.3 1.2-3.5 3.5-.2-2.3-1.2-3.3-3.5-3.5 2.3-.3 3.3-1.3 3.5-3.5Z" />
    </svg>
  )
}

function ProgressRing({ progress }: { progress: number }) {
  const radius = 42
  const circumference = 2 * Math.PI * radius
  return (
    <div className="progress-ring" aria-label={`${Math.round(progress * 100)} percent complete`}>
      <svg viewBox="0 0 100 100">
        <circle className="progress-track" cx="50" cy="50" r={radius} />
        <circle
          className="progress-value"
          cx="50"
          cy="50"
          r={radius}
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - progress)}
        />
      </svg>
      <strong>{Math.round(progress * 100)}%</strong>
    </div>
  )
}

function ImageComparison({ before, after }: { before: string; after: string }) {
  const [position, setPosition] = useState(50)
  return (
    <div className="comparison" style={{ '--split': `${position}%` } as React.CSSProperties}>
      <img src={before} alt="Original upload" />
      <div className="comparison-after">
        <img src={after} alt="Enhanced result" />
      </div>
      <span className="compare-label before-label">Before</span>
      <span className="compare-label after-label">After</span>
      <div className="split-line" aria-hidden="true"><span>↔</span></div>
      <input
        aria-label="Compare original and enhanced image"
        type="range"
        min="0"
        max="100"
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
      />
    </div>
  )
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [mode, setMode] = useState<MediaKind>('image')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [scale, setScale] = useState(4)
  const [lighting, setLighting] = useState<LightingMode>('auto')
  const [job, setJob] = useState<JobResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    void getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!job || TERMINAL_STATES.has(job.status)) return undefined
    let active = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const latest = await getJob(job.id)
        if (!active) return
        setJob(latest)
        if (!TERMINAL_STATES.has(latest.status)) timer = window.setTimeout(poll, 900)
      } catch (caught) {
        if (active) setError(caught instanceof Error ? caught.message : 'Lost connection to job')
      }
    }
    timer = window.setTimeout(poll, 500)
    return () => {
      active = false
      if (timer) window.clearTimeout(timer)
    }
  }, [job?.id, job?.status])

  const chooseFile = (candidate: File | undefined) => {
    if (!candidate) return
    const detected = detectMediaKind(candidate.name)
    if (!detected) {
      setError('Choose PNG, JPG, JPEG, MP4, MOV, or WebM.')
      return
    }
    setMode(detected)
    setFile(candidate)
    setJob(null)
    setError(null)
  }

  const reset = () => {
    if (job && TERMINAL_STATES.has(job.status)) void deleteJob(job.id).catch(() => undefined)
    setFile(null)
    setJob(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const switchMode = (next: MediaKind) => {
    if (next === mode) return
    reset()
    setMode(next)
  }

  const enhance = async () => {
    if (!file) return
    setError(null)
    try {
      const submitted = await submitJob(file, mode, { scale, lighting })
      setJob(submitted)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not start enhancement')
    }
  }

  const busy = job?.status === 'queued' || job?.status === 'processing'
  const complete = job?.status === 'completed' && job.output_url
  const accepted = mode === 'image' ? '.png,.jpg,.jpeg' : '.mp4,.mov,.webm'

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="ClearFrame home">
          <Mark />
          <span>CLEAR<span>/</span>FRAME</span>
        </a>
        <div className="system-status">
          <span className={health?.cuda_available ? 'status-dot online' : 'status-dot'} />
          <span>{health?.cuda_available ? health.gpu : health ? 'CPU mode' : 'Connecting'}</span>
          <b>LOCAL</b>
        </div>
      </header>

      <main>
        <section className="intro">
          <div className="eyebrow"><span>01</span> Restoration studio</div>
          <h1>Recover what the <em>dark</em> left behind.</h1>
          <p>
            Neural detail recovery for soft, compressed, and underexposed media.
            Your files stay on this machine.
          </p>
        </section>

        <section className="studio" aria-label="Enhancement workspace">
          <div className="control-panel">
            <div className="mode-tabs" role="tablist" aria-label="Media type">
              {(['image', 'video'] as MediaKind[]).map((item) => (
                <button
                  type="button"
                  key={item}
                  role="tab"
                  aria-selected={mode === item}
                  className={mode === item ? 'active' : ''}
                  onClick={() => switchMode(item)}
                  disabled={busy}
                >
                  <span>0{item === 'image' ? 1 : 2}</span>{item}
                </button>
              ))}
            </div>

            {!file ? (
              <button
                type="button"
                className={`drop-zone ${dragging ? 'dragging' : ''}`}
                onClick={() => inputRef.current?.click()}
                onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault()
                  setDragging(false)
                  chooseFile(event.dataTransfer.files[0])
                }}
              >
                <span className="upload-icon"><UploadIcon /></span>
                <strong>Drop {mode === 'image' ? 'an image' : 'a video'} here</strong>
                <span>or browse this computer</span>
                <small>{mode === 'image' ? 'PNG · JPG · JPEG' : 'MP4 · MOV · WEBM'}</small>
              </button>
            ) : (
              <div className="file-card">
                <div className="file-kind">{mode === 'image' ? 'IMG' : 'VID'}</div>
                <div className="file-copy">
                  <strong>{file.name}</strong>
                  <span>{formatBytes(file.size)} · {file.type || mode}</span>
                </div>
                {!busy && <button type="button" onClick={reset}>Replace</button>}
              </div>
            )}
            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept={accepted}
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            {mode === 'image' ? (
              <div className="settings-grid">
                <fieldset>
                  <legend>Output scale</legend>
                  <div className="choice-row">
                    {[1, 2, 4].map((value) => (
                      <button
                        type="button"
                        key={value}
                        className={scale === value ? 'selected' : ''}
                        onClick={() => setScale(value)}
                        disabled={busy}
                      >{value}×</button>
                    ))}
                  </div>
                </fieldset>
                <fieldset>
                  <legend>Shadow recovery</legend>
                  <select
                    value={lighting}
                    onChange={(event) => setLighting(event.target.value as LightingMode)}
                    disabled={busy}
                  >
                    <option value="auto">Auto detect</option>
                    <option value="force">Always enhance</option>
                    <option value="off">Keep original light</option>
                  </select>
                </fieldset>
              </div>
            ) : (
              <div className="video-spec">
                <span>1080p</span><p>Aspect-safe H.264 · source audio · full-frame validation</p>
              </div>
            )}

            {error && <div className="error-banner" role="alert"><b>!</b><span>{error}</span></div>}
            {job?.status === 'failed' && (
              <div className="error-banner" role="alert">
                <b>!</b><span>{job.error_message || 'The enhancement could not be completed.'}</span>
              </div>
            )}

            {!complete ? (
              <button
                className="enhance-button"
                type="button"
                disabled={!file || busy}
                onClick={enhance}
              >
                <SparkIcon />
                <span>{busy ? 'Enhancing locally' : `Enhance ${mode}`}</span>
                <i>→</i>
              </button>
            ) : (
              <div className="result-actions">
                <a className="download-button" href={job.output_url ?? '#'} download>
                  Download enhanced <span>↓</span>
                </a>
                <button type="button" onClick={reset}>New file</button>
              </div>
            )}

            <div className="privacy-note">
              <span>⌁</span>
              <p><b>Private by design.</b> Processing happens on your local GPU. Nothing is uploaded to a cloud service.</p>
            </div>
          </div>

          <div className="preview-panel">
            <div className="preview-heading">
              <div><span>Live canvas</span><strong>{complete ? 'Result ready' : file ? 'Source loaded' : 'Waiting for media'}</strong></div>
              <small>{file ? outputLabel(file.name, mode) : 'REAL-ESRGAN / CUDA'}</small>
            </div>

            <div className={`canvas ${file ? 'has-media' : ''}`}>
              {!file || !preview ? (
                <div className="empty-canvas">
                  <Mark />
                  <span>Details appear here</span>
                  <div className="coordinates"><i>0,0</i><i>1080</i></div>
                </div>
              ) : complete && mode === 'image' ? (
                <ImageComparison before={preview} after={job.output_url ?? ''} />
              ) : complete && mode === 'video' ? (
                <div className="video-comparison">
                  <figure><video src={preview} controls preload="metadata" /><figcaption>Original</figcaption></figure>
                  <figure><video src={job.output_url ?? ''} controls preload="metadata" /><figcaption>Enhanced</figcaption></figure>
                </div>
              ) : (
                <div className="source-preview">
                  {mode === 'image'
                    ? <img src={preview} alt="Selected source" />
                    : <video src={preview} controls preload="metadata" />}
                  {busy && (
                    <div className="processing-overlay">
                      <div className="scanner" />
                      <ProgressRing progress={job?.progress ?? 0} />
                      <strong>{job?.message}</strong>
                      <span>{job?.stage.toUpperCase()}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="canvas-footer">
              <span><i className="legend-dot source" />Original</span>
              <span><i className="legend-dot result" />Restored</span>
              {complete && job.report_url && <a href={job.report_url}>Technical report ↗</a>}
            </div>
          </div>
        </section>

        <section className="method-strip" aria-label="Processing stages">
          <div><span>01</span><strong>Detect</strong><p>Format, light, dimensions</p></div>
          <div><span>02</span><strong>Restore</strong><p>Noise, blur, compression</p></div>
          <div><span>03</span><strong>Validate</strong><p>Decode, size, integrity</p></div>
          <aside>Built for <b>local</b> control</aside>
        </section>
      </main>

      <footer><span>ClearFrame / v0.2</span><span>Files stay here.</span></footer>
    </div>
  )
}

export default App
