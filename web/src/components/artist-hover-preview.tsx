import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { getArtistImageUrl } from "@/api";

interface ArtistHoverPreviewProps {
  dataset: string;
  artistId: string;
  side?: "left" | "right" | "top" | "bottom";
  openDelay?: number;
  children: React.ReactNode;
}

export function ArtistHoverPreview({
  dataset,
  artistId,
  side = "right",
  openDelay = 0,
  children,
}: ArtistHoverPreviewProps) {
  return (
    <HoverCard openDelay={openDelay} closeDelay={0}>
      <HoverCardTrigger asChild>{children}</HoverCardTrigger>
      <HoverCardContent side={side} className="w-96 p-0">
        <img
          src={getArtistImageUrl(dataset, artistId)}
          alt={artistId}
          className="w-full rounded-md"
        />
      </HoverCardContent>
    </HoverCard>
  );
}
