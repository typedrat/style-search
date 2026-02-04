import { useState, useEffect } from 'react'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { ArrowLeft, Trash2 } from 'lucide-react'
import {
  type Triplet,
  type SkipReason,
  getTriplets,
  deleteTriplet,
  updateTriplet,
  getArtistImageUrl,
  listDatasets,
} from '@/api'
import { ArtistHoverPreview } from '@/components/artist-hover-preview'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ScrollArea } from '@/components/ui/scroll-area'

export const Route = createFileRoute('/train/$dataset/view')({
  component: ViewTripletsPage,
})

function ViewTripletsPage() {
  const { dataset } = Route.useParams()
  const navigate = useNavigate()

  const [datasets, setDatasets] = useState<string[]>([])
  const [triplets, setTriplets] = useState<Triplet[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    listDatasets().then(setDatasets)
  }, [])

  useEffect(() => {
    setLoading(true)
    getTriplets(dataset)
      .then((t) => setTriplets([...t].reverse()))
      .finally(() => setLoading(false))
  }, [dataset])

  const handleDatasetChange = (newDataset: string) => {
    navigate({ to: '/train/$dataset/view', params: { dataset: newDataset } })
  }

  const handleDelete = async (id: number) => {
    await deleteTriplet(id)
    setTriplets((prev) => prev.filter((t) => t.id !== id))
  }

  const handleUpdate = async (
    id: number,
    update: { choice: 'A' | 'B' | null; skip_reason: SkipReason | null }
  ) => {
    const updated = await updateTriplet(id, update)
    setTriplets((prev) => prev.map((t) => (t.id === id ? updated : t)))
  }

  const skipped = triplets.filter((t) => t.choice === null)
  const judged = triplets.filter((t) => t.choice !== null)

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      <header className="border-b border-border px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <Link to="/train/$dataset" params={{ dataset }}>
            <Button variant="ghost" size="icon">
              <ArrowLeft className="size-4" />
            </Button>
          </Link>
          <h1 className="text-lg font-semibold">View Triplets</h1>
        </div>
        <div className="flex items-center gap-4">
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
          <span className="text-sm text-muted-foreground">
            {judged.length} judged, {skipped.length} skipped
          </span>
        </div>
      </header>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {loading ? (
            <div className="text-center text-muted-foreground">Loading...</div>
          ) : triplets.length === 0 ? (
            <div className="text-center text-muted-foreground">
              No triplets collected yet.{' '}
              <Link to="/train/$dataset" params={{ dataset }} className="underline">
                Start training
              </Link>
            </div>
          ) : (
            triplets.map((triplet) => (
              <TripletCard
                key={triplet.id}
                triplet={triplet}
                dataset={dataset}
                onDelete={() => triplet.id && handleDelete(triplet.id)}
                onUpdate={(update) => triplet.id && handleUpdate(triplet.id, update)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}

function displayName(id: string): string {
  return id.replace(/_\(artist\)$/i, '')
}

const SKIP_REASON_LABELS: Record<SkipReason, string> = {
  too_similar: 'Too similar',
  anchor_outlier: 'Anchor too different',
  unknown: "Don't know",
}

function TripletCard({
  triplet,
  dataset,
  onDelete,
  onUpdate,
}: {
  triplet: Triplet
  dataset: string
  onDelete: () => void
  onUpdate: (update: { choice: 'A' | 'B' | null; skip_reason: SkipReason | null }) => void
}) {
  const isSkipped = triplet.choice === null

  const handleChooseA = () => {
    onUpdate({ choice: 'A', skip_reason: null })
  }

  const handleChooseB = () => {
    onUpdate({ choice: 'B', skip_reason: null })
  }

  const handleSkip = () => {
    onUpdate({ choice: null, skip_reason: 'unknown' })
  }

  const handleSkipReasonChange = (reason: string) => {
    onUpdate({ choice: null, skip_reason: reason as SkipReason })
  }

  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-ctp-surface0">
      {/* Option A */}
      <button
        onClick={handleChooseA}
        className={`flex flex-col items-center p-2 rounded-lg transition-all hover:bg-ctp-surface1 ${
          triplet.choice === 'A' ? 'ring-2 ring-ctp-green' : ''
        }`}
      >
        <ArtistHoverPreview dataset={dataset} artistId={triplet.option_a} side="top">
          <img
            src={getArtistImageUrl(dataset, triplet.option_a)}
            alt={triplet.option_a}
            className="w-24 rounded-md cursor-pointer"
          />
        </ArtistHoverPreview>
        <span className="text-xs mt-1 text-center">{displayName(triplet.option_a)}</span>
      </button>

      {/* Anchor - click to skip */}
      <button
        onClick={handleSkip}
        className="flex flex-col items-center p-2 rounded-lg transition-all hover:bg-ctp-surface1"
        title="Click to mark as skipped"
      >
        <ArtistHoverPreview dataset={dataset} artistId={triplet.anchor} side="top">
          <img
            src={getArtistImageUrl(dataset, triplet.anchor)}
            alt={triplet.anchor}
            className="w-24 rounded-md ring-2 ring-ctp-peach cursor-pointer"
          />
        </ArtistHoverPreview>
        <span className="text-xs mt-1 text-center">{displayName(triplet.anchor)}</span>
      </button>

      {/* Option B */}
      <button
        onClick={handleChooseB}
        className={`flex flex-col items-center p-2 rounded-lg transition-all hover:bg-ctp-surface1 ${
          triplet.choice === 'B' ? 'ring-2 ring-ctp-green' : ''
        }`}
      >
        <ArtistHoverPreview dataset={dataset} artistId={triplet.option_b} side="top">
          <img
            src={getArtistImageUrl(dataset, triplet.option_b)}
            alt={triplet.option_b}
            className="w-24 rounded-md cursor-pointer"
          />
        </ArtistHoverPreview>
        <span className="text-xs mt-1 text-center">{displayName(triplet.option_b)}</span>
      </button>

      {/* Status / Skip reason */}
      <div className="ml-auto flex items-center gap-3">
        {isSkipped ? (
          <Select
            value={triplet.skip_reason ?? 'unknown'}
            onValueChange={handleSkipReasonChange}
          >
            <SelectTrigger className="w-40 h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(SKIP_REASON_LABELS).map(([value, label]) => (
                <SelectItem key={value} value={value}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        ) : (
          <span className="text-sm text-ctp-green">Chose {triplet.choice}</span>
        )}
        <Button variant="ghost" size="icon" onClick={onDelete}>
          <Trash2 className="size-4 text-ctp-red" />
        </Button>
      </div>
    </div>
  )
}
