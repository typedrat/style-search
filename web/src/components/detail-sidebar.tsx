import { useMemo } from "react";
import { ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArtistHoverPreview } from "@/components/artist-hover-preview";
import { getArtistImageUrl, type Artist } from "@/api";

function displayName(id: string): string {
  return id.replace(/_\(artist\)$/i, "");
}

function e621Url(id: string): string {
  return `https://e621.net/posts?tags=${encodeURIComponent(id)}`;
}

interface DetailSidebarProps {
  dataset: string;
  artist: Artist;
  similarArtists: Artist[];
  onSelectArtist: (artist: Artist) => void;
}

function SimilarityScore({ distance, min, max }: { distance: number; min: number; max: number }) {
  // Normalize distance to 0-1 range (0 = most similar, 1 = least similar)
  const range = max - min;
  const normalized = range > 0 ? (distance - min) / range : 0;

  // Color classes based on similarity
  // Most similar (low distance) = green, least similar (high distance) = red
  let colorClass: string;
  if (normalized < 0.33) {
    colorClass = "bg-ctp-green";
  }
  else if (normalized < 0.66) {
    colorClass = "bg-ctp-yellow";
  }
  else {
    colorClass = "bg-ctp-red";
  }

  return (
    <span className={`text-xs font-mono px-1.5 pt-1 pb-0.5 rounded text-ctp-crust ${colorClass}`}>
      {distance.toFixed(3)}
    </span>
  );
}

export function DetailSidebar({
  dataset,
  artist,
  similarArtists,
  onSelectArtist,
}: DetailSidebarProps) {
  const { minDistance, maxDistance } = useMemo(() => {
    const distances = similarArtists
      .map(a => a.distance)
      .filter((d): d is number => d !== undefined);
    return {
      minDistance: Math.min(...distances),
      maxDistance: Math.max(...distances),
    };
  }, [similarArtists]);

  return (
    <aside className="w-96 border-l border-border bg-card flex flex-col">
      <Card className="border-0 rounded-none shrink-0">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">
              {displayName(artist.id)}
            </CardTitle>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              asChild
            >
              <a
                href={e621Url(artist.id)}
                target="_blank"
                rel="noopener noreferrer"
                title="View on e621"
              >
                <ExternalLink className="h-4 w-4" />
              </a>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <img
            src={getArtistImageUrl(dataset, artist.id)}
            alt={artist.id}
            className="w-full rounded-md"
          />
        </CardContent>
      </Card>

      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-3 pt-4">
          Similar Artists
        </h3>
        <div className="space-y-1">
          {similarArtists.map(a => (
            <ArtistHoverPreview
              key={a.id}
              dataset={dataset}
              artistId={a.id}
              side="left"
            >
              <button
                onClick={() => onSelectArtist(a)}
                className="w-full flex items-center gap-3 p-2 rounded-md
                  hover:bg-accent hover:text-ctp-base transition-colors"
              >
                <img
                  src={getArtistImageUrl(dataset, a.id)}
                  alt={a.id}
                  className="w-12 h-12 object-cover rounded"
                />
                <span className="flex-1 text-sm text-left truncate">
                  {displayName(a.id)}
                </span>
                {a.distance !== undefined && (
                  <SimilarityScore
                    distance={a.distance}
                    min={minDistance}
                    max={maxDistance}
                  />
                )}
              </button>
            </ArtistHoverPreview>
          ))}
        </div>
      </div>
    </aside>
  );
}
