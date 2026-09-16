export type ApiStatus = 'ok' | 'fallback' | 'unavailable' | string;

export interface ApiEnvelope<T> {
  data: T;
  status: ApiStatus;
  meta?: {
    model_version?: string;
    computed_at?: string;
    environment?: string;
  };
}

const API_BASE_URL = (import.meta.env.VITE_FICOS_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

export async function apiGet<T>(
  path: string,
  params: Record<string, string | number | boolean | null | undefined> = {},
  signal?: AbortSignal
): Promise<ApiEnvelope<T>> {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      searchParams.set(key, String(value));
    }
  });

  const query = searchParams.toString();
  const response = await fetch(`${API_BASE_URL}${path}${query ? `?${query}` : ''}`, {
    signal,
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`FICOS API ${response.status}: ${response.statusText}`);
  }

  return response.json();
}
