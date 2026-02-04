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
