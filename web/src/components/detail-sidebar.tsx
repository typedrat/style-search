import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { Artist } from '@/api'

interface DetailSidebarProps {
  artist: Artist
  similarArtists: Artist[]
  onSelectArtist: (artist: Artist) => void
}

function SimilarityScore({ distance, min, max }: { distance: number; min: number; max: number }) {
  // Normalize distance to 0-1 range (0 = most similar, 1 = least similar)
  const range = max - min
  const normalized = range > 0 ? (distance - min) / range : 0

  // Color classes based on similarity
  // Most similar (low distance) = green, least similar (high distance) = red
  let colorClass: string
  if (normalized < 0.33) {
    colorClass = 'text-ctp-green'
  } else if (normalized < 0.66) {
    colorClass = 'text-ctp-yellow'
  } else {
    colorClass = 'text-ctp-red'
  }

  return (
    <span className={`text-xs font-mono ${colorClass}`}>
      {distance.toFixed(3)}
    </span>
  )
}

export function DetailSidebar({
  artist,
  similarArtists,
  onSelectArtist,
}: DetailSidebarProps) {
  const { minDistance, maxDistance } = useMemo(() => {
    const distances = similarArtists
      .map((a) => a.distance)
      .filter((d): d is number => d !== undefined)
    return {
      minDistance: Math.min(...distances),
      maxDistance: Math.max(...distances),
    }
  }, [similarArtists])

  return (
    <aside className="w-80 border-l border-border bg-card overflow-y-auto">
      <Card className="border-0 rounded-none">
        <CardHeader className="pb-3">
          <CardTitle className="text-base">{artist.id}</CardTitle>
        </CardHeader>
        <CardContent>
          {artist.uri && (
            <img
              src={artist.uri}
              alt={artist.id}
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
              onClick={() => onSelectArtist(a)}
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
                <SimilarityScore
                  distance={a.distance}
                  min={minDistance}
                  max={maxDistance}
                />
              )}
            </button>
          ))}
        </div>
      </div>
    </aside>
  )
}
