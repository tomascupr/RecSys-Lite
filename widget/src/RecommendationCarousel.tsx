import React, { useEffect, useState } from 'react';
import useEmblaCarousel from 'embla-carousel-react';
import { cn } from './lib/utils';

export interface Recommendation {
  item_id: string;
  score: number;
  title?: string;
  image_url?: string;
  price?: number;
  category?: string;
  brand?: string;
}

export interface RecommendationCarouselProps {
  apiUrl: string;
  userId: string;
  count?: number;
  title?: string;
  className?: string;
  containerClassName?: string;
  cardClassName?: string;
  onItemClick?: (item: Recommendation) => void;
  fetchItemDetails?: (itemIds: string[]) => Promise<Record<string, Partial<Recommendation>>>;
  // For testing and Storybook
  testRecommendations?: Recommendation[];
}

export const RecommendationCarousel: React.FC<RecommendationCarouselProps> = ({
  apiUrl,
  userId,
  count = 10,
  title = 'Recommended For You',
  className = '',
  containerClassName = '',
  cardClassName = '',
  onItemClick,
  fetchItemDetails,
  testRecommendations,
}) => {
  const [emblaRef, emblaApi] = useEmblaCarousel({ 
    loop: false,
    align: 'start',
    slidesToScroll: 1,
  });
  
  const [recommendations, setRecommendations] = useState<Recommendation[]>(testRecommendations || []);
  const [isLoading, setIsLoading] = useState<boolean>(!testRecommendations);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // If test recommendations are provided, use them instead of making a network request
    if (testRecommendations) {
      setRecommendations(testRecommendations);
      setIsLoading(false);
      return;
    }

    const fetchRecommendations = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(`${apiUrl}/recommend?user_id=${userId}&k=${count}`);
        
        if (!response.ok) {
          throw new Error(`Error fetching recommendations: ${response.statusText}`);
        }
        
        const data = await response.json();
        let recs = data.recommendations || [];
        
        // If fetchItemDetails is provided, get additional item details
        if (fetchItemDetails && recs.length > 0) {
          const itemIds = recs.map((rec: Recommendation) => rec.item_id);
          const details = await fetchItemDetails(itemIds);
          
          recs = recs.map((rec: Recommendation) => ({
            ...rec,
            ...details[rec.item_id],
          }));
        }
        
        setRecommendations(recs);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        console.error('Error fetching recommendations:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchRecommendations();
  }, [apiUrl, userId, count, fetchItemDetails, testRecommendations]);

  return (
    <div className={cn('w-full', className)}>
      <h2 className="text-xl font-semibold mb-4">{title}</h2>
      
      {isLoading && (
        <div className="flex justify-center items-center h-40">
          <div className="animate-spin h-8 w-8 border-4 border-gray-300 border-t-blue-600 rounded-full" />
        </div>
      )}
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}
      
      {!isLoading && !error && recommendations.length === 0 && (
        <div className="bg-gray-100 p-4 rounded text-center">
          No recommendations available.
        </div>
      )}
      
      {!isLoading && !error && recommendations.length > 0 && (
        <div className="overflow-hidden" ref={emblaRef}>
          <div className={cn('flex', containerClassName)}>
            {recommendations.map((item) => (
              <div 
                key={item.item_id}
                className={cn(
                  'flex-none w-48 mx-2 border rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow',
                  cardClassName
                )}
                onClick={() => onItemClick && onItemClick(item)}
              >
                <div className="aspect-square bg-gray-100 overflow-hidden">
                  {item.image_url ? (
                    <img 
                      src={item.image_url} 
                      alt={item.title || item.item_id}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      No Image
                    </div>
                  )}
                </div>
                <div className="p-3">
                  <h3 className="font-medium line-clamp-2">{item.title || item.item_id}</h3>
                  {item.price !== undefined && (
                    <p className="mt-1 font-bold">${item.price.toFixed(2)}</p>
                  )}
                  {item.category && (
                    <p className="mt-1 text-xs text-gray-500">{item.category}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {!isLoading && !error && recommendations.length > 0 && (
        <div className="flex justify-center mt-4 gap-2">
          <button
            onClick={() => emblaApi?.scrollPrev()}
            className="p-2 rounded-full bg-gray-200 hover:bg-gray-300"
            aria-label="Previous"
          >
            ←
          </button>
          <button
            onClick={() => emblaApi?.scrollNext()}
            className="p-2 rounded-full bg-gray-200 hover:bg-gray-300"
            aria-label="Next"
          >
            →
          </button>
        </div>
      )}
    </div>
  );
};

export default RecommendationCarousel;