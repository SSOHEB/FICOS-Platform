import { useEffect, useRef, useState } from 'react';
import * as MP4Box from 'mp4box';

interface FrameEntry {
  ts: number; // in microseconds
  blob: Blob;
}

interface UseVideoScrubOptions {
  videoSrc: string;
  containerRef: React.RefObject<HTMLElement>;
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
}

const LERP_TAU = 8;
const SNAP = 0.002;
const LRU_MAX = 24;
const LEAD = 24;
const WATCHDOG = 60000;

export function useVideoScrub({
  videoSrc,
  containerRef,
  videoRef,
  canvasRef,
}: UseVideoScrubOptions) {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [canvasLive, setCanvasLive] = useState(false);
  const [duration, setDuration] = useState(0);

  const bankRef = useRef<FrameEntry[]>([]);
  const lruRef = useRef<Map<number, ImageBitmap | null>>(new Map());
  const lruOrderRef = useRef<number[]>([]);
  
  const currentTimeRef = useRef(0);
  const targetTimeRef = useRef(0);
  const durationRef = useRef(0);
  const isReadyRef = useRef(false);
  const isRevertedRef = useRef(false);
  const isBuildingRef = useRef(false);

  // 1. Video Duration listener
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      if (video.duration && !isNaN(video.duration)) {
        durationRef.current = video.duration;
        setDuration(video.duration);
      }
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    if (video.readyState >= 1 && video.duration) {
      handleLoadedMetadata();
    }

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [videoRef]);

  // 2. Binary search for nearest frame in bank
  const findNearestFrameIndex = (targetUs: number): number => {
    const bank = bankRef.current;
    if (bank.length === 0) return -1;
    if (bank.length === 1) return 0;

    let low = 0;
    let high = bank.length - 1;

    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      if (bank[mid].ts === targetUs) return mid;
      if (bank[mid].ts < targetUs) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    if (low >= bank.length) return bank.length - 1;
    if (high < 0) return 0;

    const diffLow = Math.abs(bank[low].ts - targetUs);
    const diffHigh = Math.abs(bank[high].ts - targetUs);
    return diffLow < diffHigh ? low : high;
  };

  // 3. LRU Cache & Warmup
  const getOrLoadBitmap = async (index: number): Promise<ImageBitmap | null> => {
    const bank = bankRef.current;
    if (index < 0 || index >= bank.length) return null;

    const lru = lruRef.current;
    const lruOrder = lruOrderRef.current;

    if (lru.has(index)) {
      const idxInOrder = lruOrder.indexOf(index);
      if (idxInOrder > -1) {
        lruOrder.splice(idxInOrder, 1);
        lruOrder.push(index);
      }
      return lru.get(index) || null;
    }

    try {
      const bitmap = await createImageBitmap(bank[index].blob);
      lru.set(index, bitmap);
      lruOrder.push(index);

      // Evict oldest if exceeding capacity
      while (lruOrder.length > LRU_MAX) {
        const evictIdx = lruOrder.shift();
        if (evictIdx !== undefined && lru.has(evictIdx)) {
          const evicted = lru.get(evictIdx);
          if (evicted) evicted.close();
          lru.delete(evictIdx);
        }
      }

      return bitmap;
    } catch {
      return null;
    }
  };

  const warmSurroundingFrames = (centerIndex: number) => {
    const bank = bankRef.current;
    for (let i = centerIndex - 1; i <= centerIndex + 2; i++) {
      if (i >= 0 && i < bank.length && !lruRef.current.has(i)) {
        getOrLoadBitmap(i);
      }
    }
  };

  // 4. Main rAF loop: lerp time, draw frame or seek video
  useEffect(() => {
    let animId: number;
    let lastTime = performance.now();
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    const loop = (now: number) => {
      const deltaSeconds = Math.min(0.1, (now - lastTime) / 1000);
      lastTime = now;

      // Compute scroll progress
      const container = containerRef.current;
      let p = 0;
      if (container) {
        const totalScrollable = container.offsetHeight - window.innerHeight;
        if (totalScrollable > 0) {
          p = Math.max(0, Math.min(1, window.scrollY / totalScrollable));
        }
      }
      setScrollProgress(p);

      const dur = durationRef.current;
      if (dur > 0) {
        const target = p * dur;
        targetTimeRef.current = target;

        if (mediaQuery.matches) {
          currentTimeRef.current = target;
        } else {
          const current = currentTimeRef.current;
          let next = current + (target - current) * (1 - Math.exp(-deltaSeconds * LERP_TAU));
          if (Math.abs(target - next) < SNAP) {
            next = target;
          }
          currentTimeRef.current = next;
        }

        const t = currentTimeRef.current;

        // Draw from Frame Bank if live
        if (isReadyRef.current && !isRevertedRef.current) {
          const targetUs = t * 1_000_000;
          const frameIndex = findNearestFrameIndex(targetUs);

          if (frameIndex >= 0) {
            warmSurroundingFrames(frameIndex);
            const bitmap = lruRef.current.get(frameIndex);

            if (bitmap && canvasRef.current) {
              const canvas = canvasRef.current;
              const ctx = canvas.getContext('2d');
              if (ctx) {
                ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
                setCanvasLive(true);
              }
            } else {
              getOrLoadBitmap(frameIndex).then((loaded) => {
                if (loaded && canvasRef.current) {
                  const canvas = canvasRef.current;
                  const ctx = canvas.getContext('2d');
                  if (ctx) {
                    ctx.drawImage(loaded, 0, 0, canvas.width, canvas.height);
                    setCanvasLive(true);
                  }
                }
              });
            }
          }
        } else {
          // Video element seeking fallback
          const video = videoRef.current;
          if (video && !video.seeking && Math.abs(video.currentTime - t) > 0.01) {
            try {
              video.currentTime = t;
            } catch {
              // Ignore fast seek errors
            }
          }
        }
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [containerRef, videoRef, canvasRef]);

  // 5. Build Frame Bank with WebCodecs & MP4Box
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!('VideoDecoder' in window)) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    if (isBuildingRef.current) return;

    isBuildingRef.current = true;

    let isCancelled = false;
    let watchdogTimer: any;

    const buildBank = async () => {
      // 60s Watchdog
      watchdogTimer = setTimeout(() => {
        if (!isReadyRef.current) {
          isRevertedRef.current = true;
          setCanvasLive(false);
        }
      }, WATCHDOG);

      try {
        const response = await fetch(videoSrc);
        if (!response.ok) throw new Error('Video fetch failed');
        const arrayBuffer = await response.arrayBuffer();

        if (isCancelled) return;

        const mp4boxfile = MP4Box.createFile();
        let videoTrack: any = null;
        let decoder: VideoDecoder | null = null;
        let activeDecodeCount = 0;
        const pendingFrames: VideoFrame[] = [];
        let isProcessingFrames = false;

        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 1920;
        offscreenCanvas.height = 1080;
        const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: false });

        const processFrameQueue = async () => {
          if (isProcessingFrames) return;
          isProcessingFrames = true;

          while (pendingFrames.length > 0) {
            const frame = pendingFrames.shift()!;
            const ts = frame.timestamp;

            if (offscreenCtx) {
              offscreenCtx.drawImage(frame, 0, 0, 1920, 1080);
              frame.close();

              await new Promise<void>((resolve) => {
                offscreenCanvas.toBlob(
                  (blob) => {
                    if (blob && !isCancelled) {
                      bankRef.current.push({ ts, blob });
                    }
                    resolve();
                  },
                  'image/webp',
                  0.82
                );
              });
            } else {
              frame.close();
            }
          }

          isProcessingFrames = false;
        };

        const initDecoder = (preferSoftware = false) => {
          decoder = new VideoDecoder({
            output: (frame) => {
              activeDecodeCount--;
              pendingFrames.push(frame);
              processFrameQueue();
            },
            error: (e) => {
              if (!preferSoftware) {
                initDecoder(true);
              } else {
                isRevertedRef.current = true;
                setCanvasLive(false);
              }
            },
          });
        };

        initDecoder(false);

        mp4boxfile.onReady = (info) => {
          videoTrack = info.videoTracks[0];
          if (!videoTrack || !decoder) return;

          // Build codec string description
          let description: Uint8Array | undefined;
          const trak = mp4boxfile.getTrackById(videoTrack.id);
          if (trak && (trak as any).mdia && (trak as any).mdia.minf && (trak as any).mdia.minf.stbl && (trak as any).mdia.minf.stbl.stsd) {
            const entries = (trak as any).mdia.minf.stbl.stsd.entries;
            if (entries && entries[0]) {
              const entry = entries[0];
              const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
              if (box) {
                const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
                box.write(stream);
                description = new Uint8Array(stream.buffer, 8);
              }
            }
          }

          try {
            decoder.configure({
              codec: videoTrack.codec,
              codedWidth: videoTrack.video?.width || 1920,
              codedHeight: videoTrack.video?.height || 1080,
              description,
              hardwareAcceleration: 'no-preference',
            });

            mp4boxfile.setExtractionOptions(videoTrack.id, null, { nbSamples: 1000 });
            mp4boxfile.start();
          } catch (err) {
            isRevertedRef.current = true;
          }
        };

        mp4boxfile.onSamples = async (_id, _user, samples) => {
          if (!decoder) return;

          for (const sample of samples) {
            while (activeDecodeCount > LEAD && !isCancelled) {
              await new Promise((r) => setTimeout(r, 16));
            }

            if (isCancelled) break;

            const chunk = new EncodedVideoChunk({
              type: sample.is_sync ? 'key' : 'delta',
              timestamp: (sample.cts * 1_000_000) / sample.timescale,
              duration: (sample.duration * 1_000_000) / sample.timescale,
              data: sample.data,
            });

            activeDecodeCount++;
            decoder.decode(chunk);
          }

          if (decoder.state === 'configured') {
            await decoder.flush();
            await processFrameQueue();
            
            // Sort bank by timestamp
            bankRef.current.sort((a, b) => a.ts - b.ts);
            if (bankRef.current.length > 10) {
              isReadyRef.current = true;
              clearTimeout(watchdogTimer);
            }
          }
        };

        const buffer = arrayBuffer as ArrayBuffer & { fileStart: number };
        buffer.fileStart = 0;
        mp4boxfile.appendBuffer(buffer);
        mp4boxfile.flush();
      } catch {
        isRevertedRef.current = true;
        setCanvasLive(false);
      }
    };

    // Run after window load
    if (document.readyState === 'complete') {
      buildBank();
    } else {
      window.addEventListener('load', buildBank, { once: true });
    }

    return () => {
      isCancelled = true;
      clearTimeout(watchdogTimer);
    };
  }, [videoSrc]);

  return { scrollProgress, canvasLive, duration };
}
