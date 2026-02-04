export interface Artist {
  id: string
  distance?: number
  metadata: Record<string, unknown>
  uri?: string
}

export interface DatasetInfo {
  name: string
  count: number
  metadata: Record<string, unknown>
}

export async function listDatasets(): Promise<string[]> {
  const res = await fetch('/api/datasets')
  return res.json()
}

export async function getDatasetInfo(dataset: string): Promise<DatasetInfo> {
  const res = await fetch(`/api/datasets/${dataset}`)
  return res.json()
}

export async function getArtists(dataset: string): Promise<Artist[]> {
  const res = await fetch(`/api/datasets/${dataset}/artists`)
  return res.json()
}

export async function getUmapProjection(
  dataset: string,
  nNeighbors = 15,
  minDist = 0.1
): Promise<Record<string, [number, number]>> {
  const params = new URLSearchParams({
    n_neighbors: nNeighbors.toString(),
    min_dist: minDist.toString(),
  })
  const res = await fetch(`/api/datasets/${dataset}/umap?${params}`)
  return res.json()
}

export async function findSimilar(
  dataset: string,
  artistId: string,
  nResults = 10
): Promise<Artist[]> {
  const res = await fetch(`/api/datasets/${dataset}/similar`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ artist_id: artistId, n_results: nResults }),
  })
  return res.json()
}

export async function getDistancesFrom(
  dataset: string,
  artistId: string
): Promise<Record<string, number>> {
  const res = await fetch(`/api/datasets/${dataset}/distances/${encodeURIComponent(artistId)}`)
  return res.json()
}

export function getArtistImageUrl(dataset: string, artistId: string): string {
  return `/api/datasets/${encodeURIComponent(dataset)}/images/${encodeURIComponent(artistId)}`
}

export type SkipReason = 'too_similar' | 'anchor_outlier' | 'unknown'

export interface Triplet {
  id?: number
  dataset: string
  anchor: string
  option_a: string
  option_b: string
  choice: 'A' | 'B' | null  // null = skipped
  skip_reason: SkipReason | null
  timestamp: number
}

export async function createTriplet(triplet: Omit<Triplet, 'id'>): Promise<Triplet> {
  const res = await fetch('/api/triplets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(triplet),
  })
  return res.json()
}

export async function getTriplets(dataset?: string): Promise<Triplet[]> {
  const url = dataset ? `/api/triplets?dataset=${encodeURIComponent(dataset)}` : '/api/triplets'
  const res = await fetch(url)
  return res.json()
}

export async function updateTriplet(
  id: number,
  update: { choice: 'A' | 'B' | null; skip_reason: SkipReason | null }
): Promise<Triplet> {
  const res = await fetch(`/api/triplets/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update),
  })
  return res.json()
}

export async function deleteTriplet(id: number): Promise<void> {
  await fetch(`/api/triplets/${id}`, { method: 'DELETE' })
}

export interface SuggestedTriplet {
  anchor: string
  option_a: string
  option_b: string
  uncertainty_score: number
  diversity_score: number
}

export async function suggestTriplet(dataset: string): Promise<SuggestedTriplet> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(dataset)}/suggest-triplet`)
  if (!res.ok) {
    throw new Error(`Failed to suggest triplet: ${res.statusText}`)
  }
  return res.json()
}

export interface ModelStatus {
  loaded: boolean
  dim: number | null
  num_triplets: number
  train_accuracy: number | null
  weights_path: string | null
  weights_exist: boolean
}

export async function getModelStatus(dataset: string): Promise<ModelStatus> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(dataset)}/model-status`)
  if (!res.ok) {
    throw new Error(`Failed to get model status: ${res.statusText}`)
  }
  return res.json()
}

export interface RetrainResult {
  train_accuracy?: number
  baseline_accuracy?: number
  improvement?: number
  num_triplets: number
  dim?: number
  error?: string
}

export async function retrainModel(dataset: string): Promise<RetrainResult> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(dataset)}/retrain`, {
    method: 'POST',
  })
  if (!res.ok) {
    throw new Error(`Failed to retrain model: ${res.statusText}`)
  }
  return res.json()
}
