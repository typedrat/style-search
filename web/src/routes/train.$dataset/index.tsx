import { useState, useEffect, useCallback } from 'react'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { ArrowLeft } from 'lucide-react'
import {
  type Artist,
  type SkipReason,
  getArtists,
  getArtistImageUrl,
  getTriplets,
  createTriplet,
  listDatasets,
  suggestTriplet,
} from '@/api'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'

export const Route = createFileRoute('/train/$dataset/')({
  component: TrainView,
})

function pickRandom<T>(arr: T[], exclude: Set<T> = new Set()): T {
  const filtered = arr.filter((x) => !exclude.has(x))
  return filtered[Math.floor(Math.random() * filtered.length)]
}

function TrainView() {
  const { dataset } = Route.useParams()
  const navigate = useNavigate()

  const [datasets, setDatasets] = useState<string[]>([])
  const [artists, setArtists] = useState<Artist[]>([])
  const [tripletCount, setTripletCount] = useState(0)

  const [anchor, setAnchor] = useState<Artist | null>(null)
  const [optionA, setOptionA] = useState<Artist | null>(null)
  const [optionB, setOptionB] = useState<Artist | null>(null)

  // Suggested vs random mode
  const [useSuggested, setUseSuggested] = useState(true)
  const [suggestionScores, setSuggestionScores] = useState<{ uncertainty: number; diversity: number } | null>(null)
  const [suggestionLoading, setSuggestionLoading] = useState(false)

  // Load datasets
  useEffect(() => {
    listDatasets().then(setDatasets)
  }, [])

  // Load artists and triplet count when dataset changes
  useEffect(() => {
    getArtists(dataset).then(setArtists)
    getTriplets(dataset).then((t) => setTripletCount(t.length))
  }, [dataset])

  // Reload triplet when mode changes
  useEffect(() => {
    if (artists.length >= 3) {
      if (useSuggested) {
        pickSuggestedTriplet()
      } else {
        pickRandomTriplet()
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useSuggested])

  // Pick new triplet (random mode)
  const pickRandomTriplet = useCallback(() => {
    if (artists.length < 3) return
    const a = pickRandom(artists)
    const b = pickRandom(artists, new Set([a]))
    const c = pickRandom(artists, new Set([a, b]))
    setAnchor(a)
    setOptionA(b)
    setOptionB(c)
    setSuggestionScores(null)
  }, [artists])

  // Pick new triplet (suggested mode)
  const pickSuggestedTriplet = useCallback(async () => {
    if (artists.length < 3) return

    setSuggestionLoading(true)
    try {
      const suggested = await suggestTriplet(dataset)
      const artistMap = new Map(artists.map((a) => [a.id, a]))

      const anchorArtist = artistMap.get(suggested.anchor)
      const optionAArtist = artistMap.get(suggested.option_a)
      const optionBArtist = artistMap.get(suggested.option_b)

      if (anchorArtist && optionAArtist && optionBArtist) {
        setAnchor(anchorArtist)
        setOptionA(optionAArtist)
        setOptionB(optionBArtist)
        setSuggestionScores({
          uncertainty: suggested.uncertainty_score,
          diversity: suggested.diversity_score,
        })
      } else {
        // Fallback to random if artist not found
        pickRandomTriplet()
      }
    } catch (error) {
      console.error('Failed to get suggested triplet:', error)
      // Fallback to random
      pickRandomTriplet()
    } finally {
      setSuggestionLoading(false)
    }
  }, [artists, dataset, pickRandomTriplet])

  // Pick new triplet based on mode
  const pickNewTriplet = useCallback(() => {
    if (useSuggested) {
      pickSuggestedTriplet()
    } else {
      pickRandomTriplet()
    }
  }, [useSuggested, pickSuggestedTriplet, pickRandomTriplet])

  // Pick initial triplet when artists load
  useEffect(() => {
    if (artists.length >= 3) {
      pickNewTriplet()
    }
  }, [artists, pickNewTriplet])

  const handleChoice = useCallback(
    async (choice: 'A' | 'B' | 'skip', skipReason?: SkipReason) => {
      if (!anchor || !optionA || !optionB) return

      await createTriplet({
        dataset,
        anchor: anchor.id,
        option_a: optionA.id,
        option_b: optionB.id,
        choice: choice === 'skip' ? null : choice,
        skip_reason: choice === 'skip' ? (skipReason ?? null) : null,
        timestamp: Date.now(),
      })
      setTripletCount((c) => c + 1)

      pickNewTriplet()
    },
    [anchor, optionA, optionB, dataset, pickNewTriplet]
  )

  const handleDatasetChange = (newDataset: string) => {
    navigate({ to: '/train/$dataset', params: { dataset: newDataset } })
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') {
        handleChoice('A')
      } else if (e.key === 'b' || e.key === 'B' || e.key === 'ArrowRight') {
        handleChoice('B')
      } else if (e.key === 's' || e.key === 'S') {
        // S = too similar
        handleChoice('skip', 'too_similar')
      } else if (e.key === 'd' || e.key === 'D') {
        // D = anchor too different (outlier)
        handleChoice('skip', 'anchor_outlier')
      } else if (e.key === ' ') {
        // Space = unknown/other
        e.preventDefault()
        handleChoice('skip', 'unknown')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleChoice])

  const handleExport = async () => {
    const triplets = await getTriplets(dataset)
    const blob = new Blob([JSON.stringify(triplets, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `triplets-${dataset}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!anchor || !optionA || !optionB || suggestionLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background text-muted-foreground">
        {suggestionLoading ? 'Getting suggestion...' : 'Loading...'}
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      <header className="border-b border-border px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <Link to="/$dataset" params={{ dataset }}>
            <Button variant="ghost" size="icon">
              <ArrowLeft className="size-4" />
            </Button>
          </Link>
          <h1 className="text-lg font-semibold">Style Similarity Training</h1>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Random</span>
            <Switch
              checked={useSuggested}
              onCheckedChange={setUseSuggested}
              disabled={suggestionLoading}
            />
            <span className="text-sm text-muted-foreground">Suggested</span>
          </div>
          <Select value={dataset} onValueChange={handleDatasetChange}>
            <SelectTrigger className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {datasets.map((ds) => (
                <SelectItem key={ds} value={ds}>
                  {ds}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Link
            to="/train/$dataset/view"
            params={{ dataset }}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            {tripletCount} triplets collected
          </Link>
          <Button variant="outline" size="sm" onClick={handleExport}>
            Export
          </Button>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center gap-4 p-4 min-h-0">
        <div className="text-center shrink-0">
          <p className="text-muted-foreground">
            Which style is more similar to the anchor?
          </p>
          {suggestionScores && (
            <p className="text-xs text-muted-foreground/60 mt-1">
              Uncertainty: {suggestionScores.uncertainty.toFixed(3)} | Diversity: {suggestionScores.diversity.toFixed(3)}
            </p>
          )}
        </div>

        <div className="flex items-start gap-4 flex-1 min-h-0 max-h-full">
          {/* Option A */}
          <button
            onClick={() => handleChoice('A')}
            className="group focus:outline-none h-full"
          >
            <ArtistCard
              dataset={dataset}
              artist={optionA}
              label="A"
              clickable
            />
          </button>

          {/* Anchor */}
          <ArtistCard
            dataset={dataset}
            artist={anchor}
            label="Anchor"
            highlight
          />

          {/* Option B */}
          <button
            onClick={() => handleChoice('B')}
            className="group focus:outline-none h-full"
          >
            <ArtistCard
              dataset={dataset}
              artist={optionB}
              label="B"
              clickable
            />
          </button>
        </div>

        <div className="flex gap-2 shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleChoice('skip', 'too_similar')}
            title="Keyboard: S"
          >
            Too similar (S)
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleChoice('skip', 'anchor_outlier')}
            title="Keyboard: D"
          >
            Anchor too different (D)
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleChoice('skip', 'unknown')}
            title="Keyboard: Space"
          >
            Don't know (Space)
          </Button>
        </div>
      </main>
    </div>
  )
}

function displayName(id: string): string {
  return id.replace(/_\(artist\)$/i, '')
}

function ArtistCard({
  dataset,
  artist,
  label,
  highlight,
  clickable,
}: {
  dataset: string
  artist: Artist
  label: string
  highlight?: boolean
  clickable?: boolean
}) {
  return (
    <div
      className={`h-full flex flex-col rounded-lg border-2 transition-all ${
        highlight
          ? 'border-ctp-peach'
          : clickable
            ? 'border-transparent group-hover:border-ctp-lavender group-focus:border-ctp-lavender'
            : 'border-border'
      }`}
    >
      <img
        src={getArtistImageUrl(dataset, artist.id)}
        alt={artist.id}
        className="flex-1 min-h-0 object-contain rounded-t-md"
      />
      <div className={`p-2 bg-ctp-surface0 rounded-b-md shrink-0 ${highlight ? 'text-center' : ''}`}>
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-sm font-medium truncate">{displayName(artist.id)}</div>
      </div>
    </div>
  )
}
