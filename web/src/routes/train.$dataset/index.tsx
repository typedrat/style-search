import { useState, useEffect, useCallback, useRef } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { ArrowLeft, User } from "lucide-react";
import {
  type Artist,
  type SkipReason,
  getArtists,
  getArtistImageUrl,
  getTriplets,
  createTriplet,
  listDatasets,
  suggestTriplet,
} from "@/api";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useUser } from "@/components/user-provider";

export const Route = createFileRoute("/train/$dataset/")({
  component: TrainView,
});

function pickRandom<T>(arr: T[], exclude: Set<T> = new Set()): T {
  const filtered = arr.filter(x => !exclude.has(x));
  return filtered[Math.floor(Math.random() * filtered.length)];
}

function TrainView() {
  const { dataset } = Route.useParams();
  const navigate = useNavigate();
  const { user, loading: userLoading, error: userError } = useUser();

  const [datasets, setDatasets] = useState<string[]>([]);
  const [artists, setArtists] = useState<Artist[]>([]);
  const [tripletCount, setTripletCount] = useState(0);

  const [anchor, setAnchor] = useState<Artist | null>(null);
  const [optionA, setOptionA] = useState<Artist | null>(null);
  const [optionB, setOptionB] = useState<Artist | null>(null);

  // Suggested vs random mode
  const [useSuggested, setUseSuggested] = useState(true);
  const [suggestionScores, setSuggestionScores] = useState<{ uncertainty: number; diversity: number } | null>(null);
  const [suggestionLoading, setSuggestionLoading] = useState(false);

  // Load datasets
  useEffect(() => {
    void listDatasets().then(setDatasets);
  }, []);

  // Load artists and triplet count when dataset changes
  useEffect(() => {
    void getArtists(dataset).then(setArtists);
    void getTriplets(dataset).then(t => setTripletCount(t.length));
  }, [dataset]);

  // Reload triplet when mode changes
  useEffect(() => {
    if (artists.length >= 3) {
      if (useSuggested) {
        void pickSuggestedTriplet();
      }
      else {
        pickRandomTriplet();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [useSuggested]);

  // Pick new triplet (random mode)
  const pickRandomTriplet = useCallback(() => {
    if (artists.length < 3) return;
    const a = pickRandom(artists);
    const b = pickRandom(artists, new Set([a]));
    const c = pickRandom(artists, new Set([a, b]));
    setAnchor(a);
    setOptionA(b);
    setOptionB(c);
    setSuggestionScores(null);
  }, [artists]);

  // Pick new triplet (suggested mode)
  const pickSuggestedTriplet = useCallback(async () => {
    if (artists.length < 3) return;

    setSuggestionLoading(true);
    try {
      const suggested = await suggestTriplet(dataset);
      const artistMap = new Map(artists.map(a => [a.id, a]));

      const anchorArtist = artistMap.get(suggested.anchor);
      const optionAArtist = artistMap.get(suggested.option_a);
      const optionBArtist = artistMap.get(suggested.option_b);

      if (anchorArtist && optionAArtist && optionBArtist) {
        setAnchor(anchorArtist);
        setOptionA(optionAArtist);
        setOptionB(optionBArtist);
        setSuggestionScores({
          uncertainty: suggested.uncertainty_score,
          diversity: suggested.diversity_score,
        });
      }
      else {
        // Fallback to random if artist not found
        pickRandomTriplet();
      }
    }
    catch (error) {
      console.error("Failed to get suggested triplet:", error);
      // Fallback to random
      pickRandomTriplet();
    }
    finally {
      setSuggestionLoading(false);
    }
  }, [artists, dataset, pickRandomTriplet]);

  // Pick new triplet based on mode
  const pickNewTriplet = useCallback(() => {
    if (useSuggested) {
      void pickSuggestedTriplet();
    }
    else {
      pickRandomTriplet();
    }
  }, [useSuggested, pickSuggestedTriplet, pickRandomTriplet]);

  // Pick initial triplet when artists load
  useEffect(() => {
    if (artists.length >= 3) {
      pickNewTriplet();
    }
  }, [artists, pickNewTriplet]);

  const handleChoice = useCallback(
    async (choice: "A" | "B" | "skip", skipReason?: SkipReason) => {
      if (!anchor || !optionA || !optionB) return;

      await createTriplet({
        dataset,
        anchor: anchor.id,
        option_a: optionA.id,
        option_b: optionB.id,
        choice: choice === "skip" ? null : choice,
        skip_reason: choice === "skip" ? (skipReason ?? null) : null,
        timestamp: Date.now(),
      });
      setTripletCount(c => c + 1);

      pickNewTriplet();
    },
    [anchor, optionA, optionB, dataset, pickNewTriplet],
  );

  const handleDatasetChange = (newDataset: string) => {
    void navigate({ to: "/train/$dataset", params: { dataset: newDataset } });
  };

  // Swipe gesture support for mobile
  const pointerStart = useRef<{ x: number; y: number } | null>(null);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    pointerStart.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    if (!pointerStart.current) return;
    const dx = e.clientX - pointerStart.current.x;
    const dy = e.clientY - pointerStart.current.y;
    const THRESHOLD = 60;
    if (Math.abs(dx) > THRESHOLD && Math.abs(dx) > Math.abs(dy)) {
      void handleChoice(dx < 0 ? "A" : "B");
    }
    pointerStart.current = null;
  }, [handleChoice]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") {
        void handleChoice("A");
      }
      else if (e.key === "b" || e.key === "B" || e.key === "ArrowRight") {
        void handleChoice("B");
      }
      else if (e.key === "s" || e.key === "S") {
        // S = too similar
        void handleChoice("skip", "too_similar");
      }
      else if (e.key === "d" || e.key === "D") {
        // D = anchor too different (outlier)
        void handleChoice("skip", "anchor_outlier");
      }
      else if (e.key === " ") {
        // Space = unknown/other
        e.preventDefault();
        void handleChoice("skip", "unknown");
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleChoice]);

  const handleExport = async () => {
    const triplets = await getTriplets(dataset);
    const blob = new Blob([JSON.stringify(triplets, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `triplets-${dataset}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Show loading state
  if (userLoading) {
    return (
      <div className="flex items-center justify-center h-dvh bg-background text-muted-foreground">
        Checking authentication...
      </div>
    );
  }

  // Require authentication for training
  if (!user) {
    return (
      <div className="flex flex-col items-center justify-center h-dvh bg-background text-foreground gap-4">
        <h1 className="text-xl font-semibold">Authentication Required</h1>
        <p className="text-muted-foreground">
          {userError || "You need a valid user token to contribute training data."}
        </p>
        <p className="text-sm text-muted-foreground">
          Ask for a link with your user token to get started.
        </p>
        <Link to="/$dataset" params={{ dataset }}>
          <Button variant="outline">Back to Browse</Button>
        </Link>
      </div>
    );
  }

  if (!anchor || !optionA || !optionB || suggestionLoading) {
    return (
      <div className="flex items-center justify-center h-dvh bg-background text-muted-foreground">
        {suggestionLoading ? "Getting suggestion..." : "Loading..."}
      </div>
    );
  }

  return (
    <div className="h-dvh flex flex-col bg-background text-foreground">
      <header className="border-b border-border px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <Link to="/$dataset" params={{ dataset }}>
            <Button variant="ghost" size="icon">
              <ArrowLeft className="size-4" />
            </Button>
          </Link>
          <h1 className="text-lg font-semibold hidden sm:block">Style Similarity Training</h1>
        </div>
        <div className="flex items-center gap-2 sm:gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground hidden sm:inline">Random</span>
            <Switch
              checked={useSuggested}
              onCheckedChange={setUseSuggested}
              disabled={suggestionLoading}
            />
            <span className="text-sm text-muted-foreground hidden sm:inline">Suggested</span>
          </div>
          <Link
            to="/train/$dataset/view"
            params={{ dataset }}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            {tripletCount}
            {" "}
            <span className="hidden sm:inline">triplets collected</span>
          </Link>
          <div className="hidden sm:flex items-center gap-4">
            <Select value={dataset} onValueChange={handleDatasetChange}>
              <SelectTrigger className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {datasets.map(ds => (
                  <SelectItem key={ds} value={ds}>
                    {ds}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={() => void handleExport()}>
              Export
            </Button>
            <div className="flex items-center gap-1.5 text-sm text-muted-foreground border-l border-border pl-4">
              <User className="size-4" />
              <span>{user.name}</span>
            </div>
          </div>
        </div>
      </header>

      <main
        className="flex-1 flex flex-col items-center justify-center gap-4 p-4 min-h-0 touch-pan-y"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
      >
        {/* Desktop: full question text */}
        <div className="text-center shrink-0 hidden sm:block">
          <p className="text-muted-foreground">
            Which style is more similar to the anchor?
          </p>
          {suggestionScores && (
            <p className="text-xs text-muted-foreground/60 mt-1">
              Uncertainty:
              {" "}
              {suggestionScores.uncertainty.toFixed(3)}
              {" | "}
              Diversity:
              {" "}
              {suggestionScores.diversity.toFixed(3)}
            </p>
          )}
        </div>

        {/* Desktop: three cards in a row */}
        <div className="hidden sm:flex items-start gap-4 flex-1 min-h-0 max-h-full">
          <button
            onClick={() => void handleChoice("A")}
            className="group focus:outline-none h-full"
          >
            <ArtistCard
              dataset={dataset}
              artist={optionA}
              label="A"
              clickable
            />
          </button>

          <ArtistCard
            dataset={dataset}
            artist={anchor}
            label="Anchor"
            highlight
          />

          <button
            onClick={() => void handleChoice("B")}
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

        {/* Mobile: anchor small on top, A/B large below */}
        <div className="sm:hidden flex flex-col flex-1 min-h-0 w-full gap-2">
          <div className="text-center text-xs text-muted-foreground">
            Which is more similar to the anchor?
          </div>
          <div className="shrink-0 flex justify-center">
            <div className="w-1/2 max-w-56 border-2 border-ctp-peach rounded-lg">
              <img
                src={getArtistImageUrl(dataset, anchor.id)}
                alt={anchor.id}
                className="w-full rounded-t-md"
              />
              <div className="p-1 bg-ctp-surface0 text-center">
                <div className="text-xs font-medium truncate">
                  {displayName(anchor.id)}
                </div>
              </div>
            </div>
          </div>
          <div className="flex gap-3 flex-1 min-h-0 w-full">
            <button
              onClick={() => void handleChoice("A")}
              className="group focus:outline-none flex-1 min-w-0 h-full"
            >
              <ArtistCard
                dataset={dataset}
                artist={optionA}
                label="A"
                clickable
              />
            </button>
            <button
              onClick={() => void handleChoice("B")}
              className="group focus:outline-none flex-1 min-w-0 h-full"
            >
              <ArtistCard
                dataset={dataset}
                artist={optionB}
                label="B"
                clickable
              />
            </button>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-2 shrink-0 w-full sm:w-auto px-2 sm:px-0">
          <Button
            variant="outline"
            className="w-full sm:w-auto"
            onClick={() => void handleChoice("skip", "too_similar")}
            title="Keyboard: S"
          >
            Too similar
            <span className="hidden sm:inline">(S)</span>
          </Button>
          <Button
            variant="outline"
            className="w-full sm:w-auto"
            onClick={() => void handleChoice("skip", "anchor_outlier")}
            title="Keyboard: D"
          >
            Anchor too different
            <span className="hidden sm:inline">(D)</span>
          </Button>
          <Button
            variant="outline"
            className="w-full sm:w-auto"
            onClick={() => void handleChoice("skip", "unknown")}
            title="Keyboard: Space"
          >
            Don't know
            <span className="hidden sm:inline">(Space)</span>
          </Button>
        </div>
      </main>
    </div>
  );
}

function displayName(id: string): string {
  return id.replace(/_\(artist\)$/i, "");
}

function ArtistCard({
  dataset,
  artist,
  label,
  highlight,
  clickable,
}: {
  dataset: string;
  artist: Artist;
  label: string;
  highlight?: boolean;
  clickable?: boolean;
}) {
  return (
    <div
      className={`h-full flex flex-col rounded-lg border-2 transition-all ${
        highlight
          ? "border-ctp-peach"
          : clickable
            ? "border-transparent group-hover:border-ctp-lavender group-focus:border-ctp-lavender"
            : "border-border"
      }`}
    >
      <img
        src={getArtistImageUrl(dataset, artist.id)}
        alt={artist.id}
        className="flex-1 min-h-0 object-contain rounded-t-md"
      />
      <div className={`p-2 bg-ctp-surface0 rounded-b-md shrink-0 ${highlight ? "text-center" : ""}`}>
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-sm font-medium truncate">{displayName(artist.id)}</div>
      </div>
    </div>
  );
}
