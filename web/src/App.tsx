import { useEffect, useState } from 'react'
import Scatterplot from './Scatterplot'
import {
  type Artist,
  findSimilar,
  getArtists,
  getDistancesFrom,
  getUmapProjection,
  listDatasets,
} from './api'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ThemeSwitcher } from '@/components/theme-switcher'

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
  const [searchQuery, setSearchQuery] = useState('')

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

  const filteredArtists = searchQuery
    ? artists
        .filter((a) => a.id.toLowerCase().includes(searchQuery.toLowerCase()))
        .slice(0, 50)
    : []

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-3 border-b border-border bg-card">
        <h1 className="text-lg font-semibold">Style Search</h1>
        <Select
          value={currentDataset ?? ''}
          onValueChange={(v) => setCurrentDataset(v || null)}
          disabled={loading}
        >
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Select dataset..." />
          </SelectTrigger>
          <SelectContent>
            {datasets.map((d) => (
              <SelectItem key={d} value={d}>
                {d}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="ml-auto">
          <ThemeSwitcher />
        </div>
      </header>

      {error && (
        <div className="px-4 py-3 bg-destructive/20 text-destructive">
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Search Sidebar */}
        <aside className="w-60 border-r border-border bg-card flex flex-col">
          <div className="p-3">
            <Input
              type="text"
              placeholder="Search artists..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          {searchQuery && (
            <ScrollArea className="flex-1">
              <div className="px-2 pb-2">
                {filteredArtists.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => {
                      setSelectedArtist(a)
                      setSearchQuery('')
                    }}
                    className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
                      a.id === selectedArtist?.id
                        ? 'bg-primary text-primary-foreground'
                        : 'hover:bg-accent'
                    }`}
                  >
                    {a.id}
                  </button>
                ))}
              </div>
            </ScrollArea>
          )}
        </aside>

        {/* Main Content */}
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

        {/* Detail Sidebar */}
        {selectedArtist && (
          <aside className="w-80 border-l border-border bg-card overflow-y-auto">
            <Card className="border-0 rounded-none">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">{selectedArtist.id}</CardTitle>
              </CardHeader>
              <CardContent>
                {selectedArtist.uri && (
                  <img
                    src={selectedArtist.uri}
                    alt={selectedArtist.id}
                    className="w-full rounded-md"
                  />
                )}
              </CardContent>
            </Card>

            <div className="px-4 pb-4">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                Similar Artists
              </h3>
              <div className="space-y-1">
                {similarArtists.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => setSelectedArtist(a)}
                    className="w-full flex items-center gap-3 p-2 rounded-md hover:bg-accent transition-colors"
                  >
                    {a.uri && (
                      <img
                        src={a.uri}
                        alt={a.id}
                        className="w-10 h-10 object-cover rounded"
                      />
                    )}
                    <span className="flex-1 text-sm text-left truncate">
                      {a.id}
                    </span>
                    {a.distance !== undefined && (
                      <span className="text-xs text-muted-foreground font-mono">
                        {a.distance.toFixed(3)}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  )
}

export default App
