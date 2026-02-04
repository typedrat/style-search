import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useTheme } from './theme-provider'

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme()

  return (
    <Select value={theme} onValueChange={setTheme}>
      <SelectTrigger className="w-36">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="system">System</SelectItem>
        <SelectItem value="latte">Latte</SelectItem>
        <SelectItem value="frappe">Frappé</SelectItem>
        <SelectItem value="macchiato">Macchiato</SelectItem>
        <SelectItem value="mocha">Mocha</SelectItem>
      </SelectContent>
    </Select>
  )
}
