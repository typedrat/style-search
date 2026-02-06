import { useMemo, useState } from "react";
import { matchSorter } from "match-sorter";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArtistHoverPreview } from "@/components/artist-hover-preview";
import type { Artist } from "@/api";

function displayName(id: string): string {
  return id.replace(/_\(artist\)$/i, "");
}

function highlightMatches(text: string, query: string): React.ReactNode {
  if (!query) return text;

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const matches: boolean[] = new Array<boolean>(text.length).fill(false);

  // Find matching characters (fuzzy match)
  let queryIdx = 0;
  for (let i = 0; i < text.length && queryIdx < query.length; i++) {
    if (lowerText[i] === lowerQuery[queryIdx]) {
      matches[i] = true;
      queryIdx++;
    }
  }

  // Build result with highlighted spans
  const result: React.ReactNode[] = [];
  let i = 0;
  while (i < text.length) {
    if (matches[i]) {
      let end = i;
      while (end < text.length && matches[end]) end++;
      result.push(
        <span key={i} className="font-semibold">
          {text.slice(i, end)}
        </span>,
      );
      i = end;
    }
    else {
      let end = i;
      while (end < text.length && !matches[end]) end++;
      result.push(text.slice(i, end));
      i = end;
    }
  }

  return result;
}

interface SearchSidebarProps {
  dataset: string | null;
  artists: Artist[];
  selectedArtist: Artist | null;
  onSelectArtist: (artist: Artist) => void;
}

export function SearchSidebar({
  dataset,
  artists,
  selectedArtist,
  onSelectArtist,
}: SearchSidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredArtists = useMemo(() => {
    if (!searchQuery) return [];
    return matchSorter(artists, searchQuery, {
      keys: [a => displayName(a.id)],
    }).slice(0, 50);
  }, [artists, searchQuery]);

  return (
    <aside className="w-60 border-r border-border bg-card flex flex-col">
      <div className="p-3">
        <Input
          type="text"
          placeholder="Search artists..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
        />
      </div>
      {searchQuery && (
        <ScrollArea className="flex-1">
          <div className="px-2 pb-2 space-y-1">
            {filteredArtists.map(a =>
              dataset
                ? (
                    <ArtistHoverPreview
                      key={a.id}
                      dataset={dataset}
                      artistId={a.id}
                      side="right"
                    >
                      <button
                        onClick={() => onSelectArtist(a)}
                        className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
                          a.id === selectedArtist?.id
                            ? "bg-primary text-primary-foreground"
                            : "hover:bg-accent hover:text-ctp-base"
                        }`}
                      >
                        {highlightMatches(displayName(a.id), searchQuery)}
                      </button>
                    </ArtistHoverPreview>
                  )
                : (
                    <button
                      key={a.id}
                      onClick={() => onSelectArtist(a)}
                      className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
                        a.id === selectedArtist?.id
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-accent hover:text-ctp-base"
                      }`}
                    >
                      {highlightMatches(displayName(a.id), searchQuery)}
                    </button>
                  ),
            )}
          </div>
        </ScrollArea>
      )}
    </aside>
  );
}
