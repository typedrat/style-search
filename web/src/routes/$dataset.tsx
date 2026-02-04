import { useEffect, useState, useMemo } from 'react'
import { createFileRoute, useNavigate, Outlet, useParams } from '@tanstack/react-router'
import {
  type Artist,
  findSimilar,
  getArtists,
  getDistancesFrom,
  getUmapProjection,
  listDatasets,
} from '@/api'
import { Header } from '@/components/header'
import { SearchSidebar } from '@/components/search-sidebar'
import { DetailSidebar } from '@/components/detail-sidebar'
import { Scatterplot } from '@/components/scatterplot'
import { Spinner } from '@/components/ui/spinner'

export const Route = createFileRoute('/$dataset')({
  component: DatasetView,
})

function DatasetView() {
  const { dataset } = Route.useParams()
  const params = useParams({ strict: false })
  const navigate = useNavigate()

  const artistId = (params as { artistId?: string }).artistId ?? null

  const [datasets, setDatasets] = useState<string[]>([])
  const [artists, setArtists] = useState<Artist[]>([])
  const [coords, setCoords] = useState<Record<string, [number, number]>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [similarArtists, setSimilarArtists] = useState<Artist[]>([])
  const [distances, setDistances] = useState<Record<string, number>>({})

  // Derive selected artist from URL param and loaded artists
  const selectedArtist = useMemo(
    () => artists.find((a) => a.id === artistId) ?? null,
    [artists, artistId]
  )

  // Load datasets on mount
  useEffect(() => {
    listDatasets()
      .then(setDatasets)
      .catch((e) => setError(e.message))
  }, [])

  // Load artists and coords when dataset changes
  useEffect(() => {
    if (!dataset) return

    setLoading(true)
    setError(null)
    setSimilarArtists([])
    setDistances({})

    Promise.all([getArtists(dataset), getUmapProjection(dataset)])
      .then(([a, c]) => {
        setArtists(a)
        setCoords(c)
        // Clear artist from URL if not in new dataset
        if (artistId && !a.some((artist) => artist.id === artistId)) {
          navigate({ to: '/$dataset', params: { dataset } })
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [dataset, artistId, navigate])

  // Load similar artists and distances when selection changes
  useEffect(() => {
    if (!dataset || !selectedArtist) {
      setSimilarArtists([])
      setDistances({})
      return
    }

    Promise.all([
      findSimilar(dataset, selectedArtist.id, 20),
      getDistancesFrom(dataset, selectedArtist.id),
    ])
      .then(([similar, dist]) => {
        setSimilarArtists(similar)
        setDistances(dist)
      })
      .catch((e) => console.error(e))
  }, [dataset, selectedArtist])

  const handleDatasetChange = (newDataset: string | null) => {
    if (!newDataset) return
    if (artistId) {
      navigate({ to: '/$dataset/$artistId', params: { dataset: newDataset, artistId } })
    } else {
      navigate({ to: '/$dataset', params: { dataset: newDataset } })
    }
  }

  const handleSelectArtist = (artist: Artist | null) => {
    if (artist) {
      navigate({ to: '/$dataset/$artistId', params: { dataset, artistId: artist.id } })
    } else {
      navigate({ to: '/$dataset', params: { dataset } })
    }
  }

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Header
        datasets={datasets}
        currentDataset={dataset}
        onDatasetChange={handleDatasetChange}
        loading={loading}
      />

      {error && (
        <div className="px-4 py-3 bg-destructive/20 text-destructive">
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <SearchSidebar
          dataset={dataset}
          artists={artists}
          selectedArtist={selectedArtist}
          onSelectArtist={handleSelectArtist}
        />

        <main className="flex-1 relative overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Spinner className="size-8 text-muted-foreground" />
            </div>
          ) : (
            <Scatterplot
              key={dataset}
              artists={artists}
              coords={coords}
              selectedArtist={selectedArtist}
              similarArtists={similarArtists}
              distances={distances}
              onSelectArtist={handleSelectArtist}
            />
          )}
        </main>

        {selectedArtist && (
          <DetailSidebar
            dataset={dataset}
            artist={selectedArtist}
            similarArtists={similarArtists}
            onSelectArtist={handleSelectArtist}
          />
        )}
      </div>
      <Outlet />
    </div>
  )
}
