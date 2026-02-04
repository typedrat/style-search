import { useState } from 'react'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { Artist } from '@/api'

interface SearchSidebarProps {
  artists: Artist[]
  selectedArtist: Artist | null
  onSelectArtist: (artist: Artist) => void
}

export function SearchSidebar({
  artists,
  selectedArtist,
  onSelectArtist,
}: SearchSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('')

  const filteredArtists = searchQuery
    ? artists
        .filter((a) => a.id.toLowerCase().includes(searchQuery.toLowerCase()))
        .slice(0, 50)
    : []

  return (
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
                  onSelectArtist(a)
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
  )
}
