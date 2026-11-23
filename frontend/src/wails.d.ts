import type { FramePreview, JobConfig, Subtitle, WorkerState } from './types'

declare global {
  interface Window {
    go?: {
      main: {
        App: {
          GetWorkerState(): Promise<WorkerState>
          RestartWorker(): Promise<void>
          SelectVideo(): Promise<string>
          GetVideoPreview(videoPath: string, ratio: number): Promise<FramePreview>
          StartJob(config: JobConfig): Promise<string>
          CancelJob(): Promise<void>
          SaveSRT(customSubtitles: Subtitle[]): Promise<string>
          FormatSRTText(customSubtitles: Subtitle[]): Promise<string>
        }
      }
    }
    runtime?: {
      EventsOn(eventName: string, callback: (...data: unknown[]) => void): () => void
      EventsOff(eventName: string, ...additionalEventNames: string[]): void
    }
  }
}

export {}
