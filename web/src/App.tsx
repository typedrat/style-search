import { useEffect, useState } from 'react'
import {
  type Artist,
  findSimilar,
  getArtists,
  getDistancesFrom,
  getUmapProjection,
  listDatasets,
} from './api'
import { Header } from '@/components/header'
import { SearchSidebar } from '@/components/search-sidebar'
import { DetailSidebar } from '@/components/detail-sidebar'
import { Scatterplot } from '@/components/scatterplot'

function App() {
  const [datasets, setDatasets] = useState<string[]>([])
  const [currentDataset, setCurrentDataset] = useState<string | null>(null)
  const [artists, setArtists] = useState<Artist[]>([])
  const [coords, setCoords] = useState<Record<string, [number, number]>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selectedArtist, setSelectedArtist] = useState<Artist | null>(null)
  const [similarArtists, setSimilarArtists] = useState<Artist[]>([])
  const [distances, setDistances] = useState<Record<string, number>>({})

  // Load datasets on mount
  useEffect(() => {
    listDatasets()
      .then((ds) => {
        setDatasets(ds)
        if (ds.length > 0) setCurrentDataset(ds[0])
      })
      .catch((e) => setError(e.message))
  }, [])

  // Load artists and coords when dataset changes
  useEffect(() => {
    if (!currentDataset) return

    setLoading(true)
    setError(null)
    setSelectedArtist(null)
    setSimilarArtists([])
    setDistances({})

    Promise.all([getArtists(currentDataset), getUmapProjection(currentDataset)])
      .then(([a, c]) => {
        setArtists(a)
        setCoords(c)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [currentDataset])

  // Load similar artists and distances when selection changes
  useEffect(() => {
    if (!currentDataset || !selectedArtist) {
      setSimilarArtists([])
      setDistances({})
      return
    }

    Promise.all([
      findSimilar(currentDataset, selectedArtist.id, 20),
      getDistancesFrom(currentDataset, selectedArtist.id),
    ])
      .then(([similar, dist]) => {
        setSimilarArtists(similar)
        setDistances(dist)
      })
      .catch((e) => console.error(e))
  }, [currentDataset, selectedArtist])

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Header
        datasets={datasets}
        currentDataset={currentDataset}
        onDatasetChange={setCurrentDataset}
        loading={loading}
      />

      {error && (
        <div className="px-4 py-3 bg-destructive/20 text-destructive">
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <SearchSidebar
          artists={artists}
          selectedArtist={selectedArtist}
          onSelectArtist={setSelectedArtist}
        />

        <main className="flex-1 relative overflow-hidden">
          {loading ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Loading UMAP projection...
            </div>
          ) : (
            <Scatterplot
              artists={artists}
              coords={coords}
              selectedArtist={selectedArtist}
              similarArtists={similarArtists}
              distances={distances}
              onSelectArtist={setSelectedArtist}
            />
          )}
        </main>

        {selectedArtist && (
          <DetailSidebar
            artist={selectedArtist}
            similarArtists={similarArtists}
            onSelectArtist={setSelectedArtist}
          />
        )}
      </div>
    </div>
  )
}

export default App
