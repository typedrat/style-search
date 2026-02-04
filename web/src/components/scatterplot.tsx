import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { Artist } from '@/api'

interface Node extends d3.SimulationNodeDatum {
  id: string
  artist: Artist
  x: number
  y: number
  initialX: number
  initialY: number
  fx?: number | null
  fy?: number | null
}

interface Props {
  artists: Artist[]
  coords: Record<string, [number, number]>
  distances?: Record<string, number>
  onSelectArtist?: (artist: Artist | null) => void
  selectedArtist?: Artist | null
  similarArtists?: Artist[]
}

export function Scatterplot({
  artists,
  coords,
  distances = {},
  onSelectArtist,
  selectedArtist,
  similarArtists = [],
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  useEffect(() => {
    const handleResize = () => {
      if (svgRef.current?.parentElement) {
        const { width, height } = svgRef.current.parentElement.getBoundingClientRect()
        setDimensions({ width, height })
      }
    }
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  useEffect(() => {
    if (!svgRef.current || artists.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const { width, height } = dimensions
    const margin = 40
    const centerX = width / 2
    const centerY = height / 2

    // Build nodes from artists + coords
    const nodes: Node[] = artists
      .filter((a) => coords[a.id])
      .map((artist) => {
        const [x, y] = coords[artist.id]
        return {
          id: artist.id,
          artist,
          x,
          y,
          initialX: x,
          initialY: y,
        }
      })

    // Scale coords to fit
    const xExtent = d3.extent(nodes, (d) => d.initialX) as [number, number]
    const yExtent = d3.extent(nodes, (d) => d.initialY) as [number, number]

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([margin, width - margin])

    const yScale = d3
      .scaleLinear()
      .domain(yExtent)
      .range([margin, height - margin])

    // Apply scaling to initial positions
    nodes.forEach((n) => {
      n.x = xScale(n.initialX)
      n.y = yScale(n.initialY)
    })

    const similarIds = new Set(similarArtists.map((a) => a.id))

    // Create container with zoom
    const g = svg.append('g')

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Force simulation
    const simulation = d3
      .forceSimulation(nodes)
      .force('collision', d3.forceCollide().radius(8))
      .alphaDecay(0.02)

    if (selectedArtist && Object.keys(distances).length > 0) {
      // Scale distances to screen space
      const distanceValues = Object.values(distances).filter((d) => d > 0)
      const maxDistance = d3.max(distanceValues) ?? 1
      const maxRadius = Math.min(width, height) / 2 - margin

      const radiusScale = d3
        .scaleSqrt()
        .domain([0, maxDistance])
        .range([0, maxRadius])

      // Initialize positions based on distance + random angle
      nodes.forEach((n) => {
        if (n.id === selectedArtist.id) {
          n.x = centerX
          n.y = centerY
          n.fx = centerX
          n.fy = centerY
        } else {
          const dist = distances[n.id] ?? maxDistance
          const radius = radiusScale(dist)
          const angle = Math.random() * 2 * Math.PI
          n.x = centerX + radius * Math.cos(angle)
          n.y = centerY + radius * Math.sin(angle)
        }
      })

      // Gentle pull toward target radius, let collision do the spreading
      simulation.force(
        'x',
        d3.forceX<Node>((d) => {
          if (d.id === selectedArtist.id) return centerX
          const dist = distances[d.id] ?? maxDistance
          const radius = radiusScale(dist)
          const angle = Math.atan2(d.y - centerY, d.x - centerX)
          return centerX + radius * Math.cos(angle)
        }).strength(0.05)
      )
      simulation.force(
        'y',
        d3.forceY<Node>((d) => {
          if (d.id === selectedArtist.id) return centerY
          const dist = distances[d.id] ?? maxDistance
          const radius = radiusScale(dist)
          const angle = Math.atan2(d.y - centerY, d.x - centerX)
          return centerY + radius * Math.sin(angle)
        }).strength(0.05)
      )

      // Collision spreads nodes apart
      simulation.force('collision', d3.forceCollide().radius(10).strength(1))
      simulation.force('charge', d3.forceManyBody().strength(-3))

      // Reset zoom
      svg.call(zoom.transform, d3.zoomIdentity)
    } else {
      // No selection: drift back to UMAP positions
      simulation.force('charge', d3.forceManyBody().strength(-2))
      simulation.force(
        'x',
        d3.forceX<Node>((d) => xScale(d.initialX)).strength(0.1)
      )
      simulation.force(
        'y',
        d3.forceY<Node>((d) => yScale(d.initialY)).strength(0.1)
      )
    }

    // Draw nodes
    const nodeElements = g
      .selectAll<SVGCircleElement, Node>('circle')
      .data(nodes, (d) => d.id)
      .join('circle')
      .attr('r', (d) =>
        d.id === selectedArtist?.id ? 10 : similarIds.has(d.id) ? 7 : 5
      )
      .attr('fill', (d) =>
        d.id === selectedArtist?.id
          ? '#f59e0b'
          : similarIds.has(d.id)
            ? '#3b82f6'
            : '#6b7280'
      )
      .attr('stroke', (d) => (d.id === selectedArtist?.id ? '#fff' : 'none'))
      .attr('stroke-width', 2)
      .attr('cursor', 'pointer')
      .on('click', (_, d) => {
        onSelectArtist?.(d.id === selectedArtist?.id ? null : d.artist)
      })

    // Tooltip
    const tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0,0,0,0.8)')
      .style('color', 'white')
      .style('padding', '8px 12px')
      .style('border-radius', '4px')
      .style('font-size', '14px')
      .style('pointer-events', 'none')
      .style('opacity', 0)
      .style('z-index', 1000)

    nodeElements
      .on('mouseenter', (event, d) => {
        tooltip
          .style('opacity', 1)
          .html(d.id)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 10 + 'px')
      })
      .on('mousemove', (event) => {
        tooltip
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 10 + 'px')
      })
      .on('mouseleave', () => {
        tooltip.style('opacity', 0)
      })

    simulation.on('tick', () => {
      nodeElements.attr('cx', (d) => d.x).attr('cy', (d) => d.y)
    })

    return () => {
      simulation.stop()
      tooltip.remove()
    }
  }, [artists, coords, dimensions, distances, selectedArtist, similarArtists, onSelectArtist])

  return (
    <svg
      ref={svgRef}
      width={dimensions.width}
      height={dimensions.height}
      style={{ display: 'block' }}
    />
  )
}
