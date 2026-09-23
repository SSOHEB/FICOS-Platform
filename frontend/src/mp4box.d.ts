declare module 'mp4box' {
  export interface MP4Track {
    id: number;
    name: string;
    codec: string;
    video?: {
      width: number;
      height: number;
    };
    duration: number;
    timescale: number;
    nb_samples: number;
  }

  export interface MP4Info {
    tracks: MP4Track[];
    videoTracks: MP4Track[];
    audioTracks: MP4Track[];
    duration: number;
    timescale: number;
    isFragmented: boolean;
    isProgressive: boolean;
    hasMoov: boolean;
  }

  export interface MP4Sample {
    track_id: number;
    number: number;
    data: Uint8Array;
    size: number;
    dts: number;
    pts: number;
    cts: number;
    duration: number;
    is_sync: boolean;
    is_rap: boolean;
    timescale: number;
    description: any;
  }

  export interface MP4File {
    onReady?: (info: MP4Info) => void;
    onError?: (e: string) => void;
    onSamples?: (id: number, user: any, samples: MP4Sample[]) => void;
    appendBuffer(data: ArrayBuffer & { fileStart: number }): number;
    flush(): void;
    setExtractionOptions(id: number, user?: any, options?: { nbSamples?: number; rapAlignment?: boolean }): void;
    start(): void;
    stop(): void;
    getTrackById(id: number): MP4Track | null;
  }

  export function createFile(): MP4File;
  
  export namespace DataStream {
    export const BIG_ENDIAN: boolean;
    export const LITTLE_ENDIAN: boolean;
  }
}
