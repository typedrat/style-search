import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ThemeSwitcher } from '@/components/theme-switcher'

interface HeaderProps {
  datasets: string[]
  currentDataset: string | null
  onDatasetChange: (dataset: string | null) => void
  loading?: boolean
}

export function Header({
  datasets,
  currentDataset,
  onDatasetChange,
  loading,
}: HeaderProps) {
  return (
    <header className="flex items-center gap-4 px-4 py-3 border-b border-border bg-card">
      <h1 className="text-lg font-semibold">Style Search</h1>
      <Select
        value={currentDataset ?? ''}
        onValueChange={(v) => onDatasetChange(v || null)}
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
  )
}
