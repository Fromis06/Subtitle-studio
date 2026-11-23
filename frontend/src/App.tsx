import { useEffect, useMemo, useRef, useState } from 'react'
import ReactCrop, { type PercentCrop } from 'react-image-crop'
import type { FramePreview, JobConfig, Subtitle, WorkerEvent, WorkerState } from './types'

const CROP_PRESETS: Record<string, PercentCrop> = {
  bottom: { unit: '%', x: 8, y: 72, width: 84, height: 22 },
  wide: { unit: '%', x: 3, y: 68, width: 94, height: 27 },
  top: { unit: '%', x: 8, y: 5, width: 84, height: 20 },
  full: { unit: '%', x: 5, y: 10, width: 90, height: 80 },
}

const DEFAULT_WORKER: WorkerState = {
  status: 'starting',
  message: 'Starting Python worker…',
}

function getAppApi() {
  const api = window.go?.main?.App
  if (!api) throw new Error('Backend bridge is not connected')
  return api
}

function readableError(error: unknown) {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  return 'Unexpected error'
}

function formatTimecode(seconds: number, includeMillis = false) {
  if (!Number.isFinite(seconds)) return includeMillis ? '00:00.00' : '00:00'
  const clamped = Math.max(0, seconds)
  const totalSeconds = Math.floor(clamped)
  const centis = Math.floor((clamped - totalSeconds) * 100)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const secs = totalSeconds % 60
  const base =
    hours > 0
      ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
      : `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  return includeMillis ? `${base}.${String(centis).padStart(2, '0')}` : base
}

function fileName(path: string) {
  return path.split(/[\\/]/).pop() || path
}

function App() {
  const [worker, setWorker] = useState<WorkerState>(DEFAULT_WORKER)
  const [videoPath, setVideoPath] = useState('')
  const [preview, setPreview] = useState<FramePreview | null>(null)
  const [previewRatio, setPreviewRatio] = useState(0.5)
  const [crop, setCrop] = useState<PercentCrop>(CROP_PRESETS.bottom)

  const [scanStep, setScanStep] = useState(1.0)
  const [boundaryStep, setBoundaryStep] = useState(0.1)
  const [maxSubtitleSeconds, setMaxSubtitleSeconds] = useState(10)
  const [minConfidence, setMinConfidence] = useState(0.4)
  const [language, setLanguage] = useState('auto')
  const [preprocessMode, setPreprocessMode] = useState('natural')

  const [running, setRunning] = useState(false)
  const [loadingPreview, setLoadingPreview] = useState(false)
  const [progress, setProgress] = useState(0)
  const [subtitles, setSubtitles] = useState<Subtitle[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [activeSubIndex, setActiveSubIndex] = useState<number | null>(null)
  const [copied, setCopied] = useState(false)
  const [notice, setNotice] = useState('No video loaded')
  const [error, setError] = useState('')
  const listEndRef = useRef<HTMLDivElement | null>(null)
  const previewSeqRef = useRef(0)

  useEffect(() => {
    getAppApi()
      .GetWorkerState()
      .then(setWorker)
      .catch((err) => {
        setWorker({ status: 'error', message: readableError(err) })
      })

    const runtime = window.runtime
    if (!runtime) return

    const unsubscribers = [
      runtime.EventsOn('worker:status', (data) => setWorker(data as WorkerState)),
      runtime.EventsOn('job:started', () => {
        setRunning(true)
        setNotice('Scanning frames and running OCR batches…')
      }),
      runtime.EventsOn('job:progress', (data) => {
        const event = data as WorkerEvent
        setProgress(event.percent ?? 0)
        if (event.current_sub) {
          setSubtitles((current) => {
            const exists = current.some(
              (item) =>
                item.start === event.current_sub?.start && item.end === event.current_sub?.end,
            )
            return exists ? current : [...current, event.current_sub as Subtitle]
          })
        }
      }),
      runtime.EventsOn('job:done', (data) => {
        const event = data as WorkerEvent
        setRunning(false)
        setProgress(100)
        const finalSubs = event.data ?? []
        setSubtitles(finalSubs)
        setNotice(`Done · ${finalSubs.length} subtitle segments extracted`)
      }),
      runtime.EventsOn('job:cancelled', () => {
        setRunning(false)
        setNotice('Job cancelled by user')
      }),
      runtime.EventsOn('job:error', (data) => {
        const event = data as WorkerEvent
        setRunning(false)
        setError(event.message || 'Failed to process video')
        setNotice('Extraction failed')
      }),
    ]

    return () => unsubscribers.forEach((unsubscribe) => unsubscribe?.())
  }, [])

  useEffect(() => {
    if (running && listEndRef.current) {
      listEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [subtitles.length, running])

  const cropStats = useMemo(() => {
    const x1 = Math.round(crop.x)
    const y1 = Math.round(crop.y)
    const wPct = Math.round(crop.width)
    const hPct = Math.round(crop.height)
    if (!preview) {
      return { pct: `X:${x1}% Y:${y1}% (${wPct}×${hPct}%)`, px: '' }
    }
    const pxW = Math.round((crop.width / 100) * preview.width)
    const pxH = Math.round((crop.height / 100) * preview.height)
    return {
      pct: `X:${x1}% Y:${y1}%`,
      px: `${pxW}×${pxH} px`,
    }
  }, [crop, preview])

  const filteredSubtitles = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    return subtitles
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => !query || item.text.toLowerCase().includes(query))
  }, [subtitles, searchQuery])

  async function loadPreview(path: string, ratio: number) {
    const clampedRatio = Math.max(0, Math.min(1, ratio))
    const seq = ++previewSeqRef.current
    setPreviewRatio(clampedRatio)
    setLoadingPreview(true)
    setError('')
    try {
      const frame = await getAppApi().GetVideoPreview(path, clampedRatio)
      if (seq !== previewSeqRef.current) return
      setPreview(frame)
      setNotice(`Frame @ ${formatTimecode(frame.timestamp, true)}`)
    } catch (err) {
      if (seq !== previewSeqRef.current) return
      setError(readableError(err))
    } finally {
      if (seq === previewSeqRef.current) {
        setLoadingPreview(false)
      }
    }
  }

  async function chooseVideo() {
    setError('')
    try {
      const path = await getAppApi().SelectVideo()
      if (!path) return
      setVideoPath(path)
      setSubtitles([])
      setActiveSubIndex(null)
      setProgress(0)
      await loadPreview(path, 0.35)
    } catch (err) {
      setError(readableError(err))
    }
  }

  async function stepTimeline(deltaSeconds: number) {
    if (!videoPath || !preview || preview.duration <= 0 || running || loadingPreview) return
    const currentSec = previewRatio * preview.duration
    const nextSec = Math.max(0, Math.min(preview.duration, currentSec + deltaSeconds))
    await loadPreview(videoPath, nextSec / preview.duration)
  }

  async function jumpToSubtitle(sub: Subtitle, index: number) {
    setActiveSubIndex(index)
    if (!videoPath || !preview || preview.duration <= 0 || running) return
    const midpoint = (sub.start + sub.end) / 2
    await loadPreview(videoPath, midpoint / preview.duration)
  }

  function applyScanPreset(preset: 'fast' | 'balanced' | 'precise') {
    if (running) return
    if (preset === 'fast') {
      setScanStep(1.5)
      setBoundaryStep(0.15)
      setMinConfidence(0.45)
    } else if (preset === 'balanced') {
      setScanStep(1.0)
      setBoundaryStep(0.1)
      setMinConfidence(0.4)
    } else {
      setScanStep(0.5)
      setBoundaryStep(0.05)
      setMinConfidence(0.35)
    }
  }

  async function startJob() {
    if (!videoPath || !preview) return
    setError('')
    setSubtitles([])
    setActiveSubIndex(null)
    setProgress(0)
    const config: JobConfig = {
      videoPath,
      crop: {
        xMin: crop.x / 100,
        xMax: (crop.x + crop.width) / 100,
        yMin: crop.y / 100,
        yMax: (crop.y + crop.height) / 100,
      },
      scanStep,
      boundaryStep,
      maxSubtitleSeconds,
      minConfidence,
      language,
      preprocessMode,
      debugTiming: false,
    }
    try {
      setRunning(true)
      setNotice('Dispatching job to OCR worker…')
      await getAppApi().StartJob(config)
    } catch (err) {
      setRunning(false)
      setError(readableError(err))
    }
  }

  async function cancelJob() {
    try {
      setNotice('Stopping active job…')
      await getAppApi().CancelJob()
    } catch (err) {
      setError(readableError(err))
    }
  }

  async function restartWorker() {
    setError('')
    try {
      await getAppApi().RestartWorker()
    } catch (err) {
      setError(readableError(err))
    }
  }

  async function saveSRT() {
    try {
      const path = await getAppApi().SaveSRT(subtitles)
      if (path) setNotice(`Exported ${fileName(path)} (${subtitles.length} lines)`)
    } catch (err) {
      setError(readableError(err))
    }
  }

  async function copySRT() {
    if (subtitles.length === 0) return
    try {
      const text = await getAppApi().FormatSRTText(subtitles)
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setNotice(`Copied ${subtitles.length} SRT entries to clipboard`)
      setTimeout(() => setCopied(false), 1800)
    } catch (err) {
      setError(readableError(err))
    }
  }

  function updateSubtitleText(index: number, text: string) {
    setSubtitles((current) =>
      current.map((item, idx) => (idx === index ? { ...item, text } : item)),
    )
  }

  function deleteSubtitle(index: number) {
    setSubtitles((current) => current.filter((_, idx) => idx !== index))
    if (activeSubIndex === index) setActiveSubIndex(null)
  }

  function mergeWithNext(index: number) {
    setSubtitles((current) => {
      if (index < 0 || index >= current.length - 1) return current
      const first = current[index]
      const second = current[index + 1]
      const mergedText =
        first.text.trim().toLowerCase() === second.text.trim().toLowerCase()
          ? first.text
          : `${first.text} ${second.text}`.trim()
      const combined: Subtitle = {
        start: first.start,
        end: Math.max(first.end, second.end),
        text: mergedText,
      }
      return [...current.slice(0, index), combined, ...current.slice(index + 2)]
    })
  }

  const workerReady = worker.status === 'ready'
  const canOpenVideo = worker.status !== 'error' && !running
  const canStart =
    workerReady && Boolean(videoPath && preview && crop.width > 1 && crop.height > 1) && !running

  return (
    <div className="studio-root">
      {/* Top Studio Header Bar */}
      <header className="studio-menubar">
        <div className="menubar-left">
          <div className="app-logo" aria-hidden="true">
            <AppLogoSvg />
          </div>
          <span className="app-title">Subtitle Studio</span>
          <span className="divider-v" />
          <button
            className="btn btn-toolbar"
            onClick={chooseVideo}
            disabled={!canOpenVideo}
            title="Open video file"
          >
            <FolderIcon />
            <span>{videoPath ? 'Open Another…' : 'Open Video…'}</span>
          </button>

          {videoPath && (
            <div className="source-badge" title={videoPath}>
              <span className="source-name">{fileName(videoPath)}</span>
              {preview && (
                <span className="source-specs">
                  {preview.width}×{preview.height} · {formatTimecode(preview.duration)}
                </span>
              )}
            </div>
          )}
        </div>

        <div className="menubar-right">
          <div className={`engine-status status-${worker.status}`} title={worker.message}>
            <span className="status-led" />
            <span className="status-text">{worker.message}</span>
            {worker.status === 'error' && (
              <button className="btn-inline-retry" onClick={restartWorker}>
                Retry
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Split Workspace */}
      <div className="studio-upper">
        {/* Left: Video Monitor & Crop Viewport */}
        <section className="monitor-pane">
          <div className="pane-toolbar">
            <div className="pane-title-group">
              <span className="pane-label">SOURCE MONITOR</span>
              {preview && (
                <span className="mono-badge">
                  ROI {cropStats.pct} {cropStats.px && `· ${cropStats.px}`}
                </span>
              )}
            </div>

            <div className="preset-bar">
              <span className="preset-caption">Crop Region:</span>
              <button
                className="btn-chip"
                onClick={() => setCrop(CROP_PRESETS.bottom)}
                disabled={!preview || running}
              >
                Bottom
              </button>
              <button
                className="btn-chip"
                onClick={() => setCrop(CROP_PRESETS.wide)}
                disabled={!preview || running}
              >
                Bottom Wide
              </button>
              <button
                className="btn-chip"
                onClick={() => setCrop(CROP_PRESETS.top)}
                disabled={!preview || running}
              >
                Top
              </button>
              <button
                className="btn-chip"
                onClick={() => setCrop(CROP_PRESETS.full)}
                disabled={!preview || running}
              >
                Full
              </button>
            </div>
          </div>

          <div className="monitor-canvas">
            {preview ? (
              <ReactCrop
                crop={crop}
                onChange={(_, percentCrop) => setCrop(percentCrop)}
                minWidth={32}
                minHeight={20}
                keepSelection
                disabled={running}
              >
                <img src={preview.image} alt="Video frame preview" draggable={false} />
              </ReactCrop>
            ) : (
              <div className="monitor-empty">
                <div className="monitor-empty-box">
                  <FrameIcon />
                  <div className="monitor-empty-text">
                    <strong>No video source loaded</strong>
                    <span>
                      Open a video file (.mp4, .mkv, .avi, .mov, .webm) and drag the bounding box
                      over the hardcoded subtitle band.
                    </span>
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={chooseVideo}
                    disabled={!canOpenVideo}
                  >
                    <FolderIcon />
                    <span>Select Video File</span>
                  </button>
                </div>
              </div>
            )}

            {loadingPreview && (
              <div className="monitor-overlay">
                <span className="loader-ring" />
                <span>Decoding frame…</span>
              </div>
            )}
          </div>

          {/* Transport Scrubber Bar */}
          <div className="transport-bar">
            <div className="transport-steppers">
              <button
                className="btn-step"
                onClick={() => stepTimeline(-5)}
                disabled={!preview || running || loadingPreview}
                title="Back 5 seconds"
              >
                -5s
              </button>
              <button
                className="btn-step"
                onClick={() => stepTimeline(-1)}
                disabled={!preview || running || loadingPreview}
                title="Back 1 second"
              >
                -1s
              </button>
              <button
                className="btn-step"
                onClick={() => stepTimeline(1)}
                disabled={!preview || running || loadingPreview}
                title="Forward 1 second"
              >
                +1s
              </button>
              <button
                className="btn-step"
                onClick={() => stepTimeline(5)}
                disabled={!preview || running || loadingPreview}
                title="Forward 5 seconds"
              >
                +5s
              </button>
            </div>

            <div className="scrubber-track-wrap">
              <input
                type="range"
                className="scrubber-input"
                min="0"
                max="1"
                step="0.002"
                value={previewRatio}
                disabled={!preview || running}
                onChange={(e) => setPreviewRatio(Number(e.target.value))}
                onMouseUp={(e) => {
                  if (videoPath) {
                    void loadPreview(videoPath, Number((e.target as HTMLInputElement).value))
                  }
                }}
                onTouchEnd={(e) => {
                  if (videoPath) {
                    void loadPreview(videoPath, Number((e.target as HTMLInputElement).value))
                  }
                }}
                onKeyUp={(e) => {
                  if (videoPath) {
                    void loadPreview(videoPath, Number((e.target as HTMLInputElement).value))
                  }
                }}
              />
            </div>

            <div className="transport-timecode">
              <span className="tc-current">
                {preview ? formatTimecode(previewRatio * preview.duration, true) : '00:00.00'}
              </span>
              <span className="tc-sep">/</span>
              <span className="tc-total">
                {preview ? formatTimecode(preview.duration, false) : '00:00'}
              </span>
              <button
                className="btn-icon"
                onClick={() => videoPath && loadPreview(videoPath, previewRatio)}
                disabled={!preview || loadingPreview || running}
                title="Reload frame at current timestamp"
              >
                <RefreshIcon />
              </button>
            </div>
          </div>
        </section>

        {/* Right: Inspector & Job Controls */}
        <aside className="inspector-pane">
          <div className="pane-toolbar">
            <span className="pane-label">OCR INSPECTOR</span>
            <div className="preset-bar">
              <button
                className="btn-chip"
                onClick={() => applyScanPreset('fast')}
                disabled={running}
                title="Faster scan for long dialogues"
              >
                Fast
              </button>
              <button
                className="btn-chip"
                onClick={() => applyScanPreset('balanced')}
                disabled={running}
                title="Default balanced parameters"
              >
                Balanced
              </button>
              <button
                className="btn-chip"
                onClick={() => applyScanPreset('precise')}
                disabled={running}
                title="High temporal precision for rapid subtitles"
              >
                Precise
              </button>
            </div>
          </div>

          <div className="inspector-body">
            <div className="param-row">
              <div className="param-top compact-top">
                <div className="param-meta">
                  <span className="param-label">OCR Language</span>
                  <span className="param-hint">PP-OCRv6 for EN/JA/ZH · EasyOCR for VI</span>
                </div>
              </div>
              <select
                className="studio-select"
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={running}
              >
                <option value="auto">Auto (EN / JA / ZH · PP-OCRv6)</option>
                <option value="ja">Japanese (PP-OCRv6)</option>
                <option value="en">English (PP-OCRv6)</option>
                <option value="vi">Vietnamese + EN (PP-OCRv6 + EasyOCR)</option>
                <option value="zh">Chinese (PP-OCRv6)</option>
                <option value="ko">Korean (PP-OCRv5)</option>
              </select>
            </div>

            <div className="param-row">
              <div className="param-top compact-top">
                <div className="param-meta">
                  <span className="param-label">Text Isolation</span>
                  <span className="param-hint">Clean noisy backgrounds if needed</span>
                </div>
              </div>
              <select
                className="studio-select"
                value={preprocessMode}
                onChange={(e) => setPreprocessMode(e.target.value)}
                disabled={running}
              >
                <option value="natural">Natural (Color + Outlined · Recommended)</option>
                <option value="clean">Isolate Bright Text (Hide Busy BG)</option>
              </select>
            </div>

            <ParamRow
              label="Scan Step"
              unit="s"
              value={scanStep}
              min={0.25}
              max={3.0}
              step={0.25}
              hint="Coarse frame sampling interval"
              disabled={running}
              onChange={setScanStep}
            />

            <ParamRow
              label="Boundary Step"
              unit="s"
              value={boundaryStep}
              min={0.05}
              max={0.5}
              step={0.05}
              hint="End-timestamp search resolution"
              disabled={running}
              onChange={setBoundaryStep}
            />

            <ParamRow
              label="Min Confidence"
              unit="%"
              value={Math.round(minConfidence * 100)}
              min={20}
              max={90}
              step={5}
              hint="Discard weak OCR detections"
              disabled={running}
              onChange={(val) => setMinConfidence(val / 100)}
            />

            <ParamRow
              label="Max Hold Time"
              unit="s"
              value={maxSubtitleSeconds}
              min={3}
              max={30}
              step={1}
              hint="Cap for a single static subtitle"
              disabled={running}
              onChange={setMaxSubtitleSeconds}
            />
          </div>

          <div className="inspector-footer">
            <div className="job-telemetry">
              <div className="telemetry-header">
                <span className="telemetry-state">
                  {running ? 'RUNNING OCR' : progress === 100 ? 'COMPLETED' : 'IDLE'}
                </span>
                <span className="telemetry-pct">{progress}%</span>
              </div>
              <div className="telemetry-bar">
                <div className="telemetry-fill" style={{ width: `${progress}%` }} />
              </div>
              <div className="telemetry-notice" title={notice}>
                {notice}
              </div>
            </div>

            <div className="job-actions">
              {running ? (
                <button className="btn btn-danger btn-block" onClick={cancelJob}>
                  <StopIcon />
                  <span>Abort Extraction</span>
                </button>
              ) : (
                <button
                  className="btn btn-primary btn-block"
                  onClick={startJob}
                  disabled={!canStart}
                >
                  <PlayIcon />
                  <span>{workerReady ? 'Run Subtitle Extraction' : 'Waiting for OCR Engine…'}</span>
                </button>
              )}
            </div>
          </div>
        </aside>
      </div>

      {/* Bottom Dock: Subtitle Timeline & Inline Editor */}
      <section className="timeline-pane">
        <div className="pane-toolbar">
          <div className="pane-title-group">
            <span className="pane-label">SUBTITLE TABLE</span>
            <span className="count-pill">{subtitles.length} rows</span>
            {subtitles.length > 0 && (
              <span className="table-tip">
                Click timecode to preview frame · Edit text inline before export
              </span>
            )}
          </div>

          <div className="timeline-tools">
            <div className="search-box">
              <SearchIcon />
              <input
                type="text"
                placeholder="Filter text…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={subtitles.length === 0}
              />
              {searchQuery && (
                <button className="search-clear" onClick={() => setSearchQuery('')}>
                  ×
                </button>
              )}
            </div>

            <button
              className="btn btn-toolbar"
              onClick={copySRT}
              disabled={subtitles.length === 0}
              title="Copy formatted SRT content to clipboard"
            >
              <CopyIcon />
              <span>{copied ? 'Copied!' : 'Copy SRT'}</span>
            </button>

            <button
              className="btn btn-toolbar btn-export"
              onClick={saveSRT}
              disabled={running || subtitles.length === 0}
              title="Export subtitles to .srt file"
            >
              <DownloadIcon />
              <span>Export .SRT</span>
            </button>
          </div>
        </div>

        <div className="table-container">
          {subtitles.length > 0 ? (
            <table className="sub-table">
              <thead>
                <tr>
                  <th className="col-idx">#</th>
                  <th className="col-tc">START</th>
                  <th className="col-tc">END</th>
                  <th className="col-dur">DUR</th>
                  <th className="col-text">RECOGNIZED TEXT (EDITABLE)</th>
                  <th className="col-act">ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {filteredSubtitles.map(({ item, index }) => {
                  const duration = Math.max(0, item.end - item.start)
                  const isActive = activeSubIndex === index
                  return (
                    <tr key={`${item.start}-${index}`} className={isActive ? 'row-active' : ''}>
                      <td className="col-idx">{String(index + 1).padStart(3, '0')}</td>
                      <td className="col-tc">
                        <button
                          className="tc-jump"
                          onClick={() => jumpToSubtitle(item, index)}
                          title="Jump preview monitor to this subtitle"
                        >
                          {formatTimecode(item.start, true)}
                        </button>
                      </td>
                      <td className="col-tc">
                        <button
                          className="tc-jump"
                          onClick={() => jumpToSubtitle(item, index)}
                          title="Jump preview monitor to this subtitle"
                        >
                          {formatTimecode(item.end, true)}
                        </button>
                      </td>
                      <td className="col-dur">{duration.toFixed(1)}s</td>
                      <td className="col-text">
                        <input
                          type="text"
                          className="sub-text-input"
                          value={item.text}
                          onChange={(e) => updateSubtitleText(index, e.target.value)}
                          onFocus={() => setActiveSubIndex(index)}
                        />
                      </td>
                      <td className="col-act">
                        <div className="row-actions">
                          {index < subtitles.length - 1 && (
                            <button
                              className="btn-row-act"
                              onClick={() => mergeWithNext(index)}
                              title="Merge with next row"
                            >
                              Merge ↓
                            </button>
                          )}
                          <button
                            className="btn-row-act btn-row-del"
                            onClick={() => deleteSubtitle(index)}
                            title="Delete row"
                          >
                            ×
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          ) : (
            <div className="table-empty">
              <span>No subtitle segments yet. Configure the crop region above and run extraction.</span>
            </div>
          )}
          <div ref={listEndRef} />
        </div>
      </section>

      {error && (
        <div className="toast-error" role="alert">
          <span className="toast-badge">ERR</span>
          <p>{error}</p>
          <button onClick={() => setError('')} aria-label="Dismiss error">
            ×
          </button>
        </div>
      )}
    </div>
  )
}

function ParamRow({
  label,
  unit,
  value,
  min,
  max,
  step,
  hint,
  disabled,
  onChange,
}: {
  label: string
  unit: string
  value: number
  min: number
  max: number
  step: number
  hint: string
  disabled: boolean
  onChange: (val: number) => void
}) {
  const formatted = step < 1 ? value.toFixed(2) : String(value)
  return (
    <div className="param-row">
      <div className="param-top">
        <div className="param-meta">
          <span className="param-label">{label}</span>
          <span className="param-hint">{hint}</span>
        </div>
        <div className="param-readout">
          {formatted}
          <span className="param-unit">{unit}</span>
        </div>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

const AppLogoSvg = () => (
  <svg viewBox="0 0 24 24" fill="none">
    <rect x="2" y="2" width="20" height="20" rx="4" fill="#181C23" stroke="#384152" strokeWidth="1.5" />
    <path d="M5 13.5V12h1.5M19 13.5V12h-1.5M5 17.5V19h1.5M19 17.5V19h-1.5" stroke="#F59E0B" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    <rect x="7.5" y="14" width="5.5" height="1.5" rx="0.75" fill="#F3F4F6" />
    <rect x="14" y="14" width="2.5" height="1.5" rx="0.75" fill="#F59E0B" />
    <rect x="8.5" y="16.5" width="7" height="1.5" rx="0.75" fill="#9CA3AF" />
    <path d="M7 7.5v2M10 6v3.5M13 5.5v4M16 7v2.5" stroke="#64748B" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
)

const FolderIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M3 6.5A1.5 1.5 0 0 1 4.5 5h4.3l2 2h8.7A1.5 1.5 0 0 1 21 8.5v9a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5v-11Z" />
  </svg>
)

const FrameIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 8V5a1 1 0 0 1 1-1h3M20 8V5a1 1 0 0 0-1-1h-3M4 16v3a1 1 0 0 0 1 1h3M20 16v3a1 1 0 0 1-1 1h-3M7 14h10M9 17h6" />
  </svg>
)

const RefreshIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M20 6v5h-5M4 18v-5h5M5.8 9A7 7 0 0 1 18 7.5L20 11M4 13l2 3.5A7 7 0 0 0 18.2 15" />
  </svg>
)

const PlayIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="m7 5 12 7-12 7V5Z" />
  </svg>
)

const StopIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="6" y="6" width="12" height="12" rx="1.5" />
  </svg>
)

const DownloadIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M12 4v11m-4.5-4.5L12 15l4.5-4.5M5 19h14" />
  </svg>
)

const CopyIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect x="9" y="9" width="11" height="11" rx="1.5" />
    <path d="M5 15H4a1.5 1.5 0 0 1-1.5-1.5v-9A1.5 1.5 0 0 1 4 3h9a1.5 1.5 0 0 1 1.5 1.5v1" />
  </svg>
)

const SearchIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <circle cx="11" cy="11" r="6.5" />
    <path d="m16 16 4.5 4.5" />
  </svg>
)

export default App
