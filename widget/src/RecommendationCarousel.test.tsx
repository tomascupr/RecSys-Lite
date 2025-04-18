/**
 * @jest-environment jsdom
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { RecommendationCarousel } from './RecommendationCarousel';

// Mock data for testing
const mockRecommendations = [
  {
    item_id: 'item1',
    score: 0.95,
    title: 'Wireless Headphones',
    category: 'Electronics',
    price: 89.99,
  },
  {
    item_id: 'item2',
    score: 0.89,
    title: 'Smart Watch',
    category: 'Electronics',
    price: 199.99,
  },
];

// Mock fetch function
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({
      user_id: 'test-user',
      recommendations: mockRecommendations,
    }),
  })
) as jest.Mock;

describe('RecommendationCarousel', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renders loading state initially', () => {
    render(
      <RecommendationCarousel
        apiUrl="https://api.example.com"
        userId="test-user"
      />
    );
    
    expect(screen.getByText('Recommended For You')).toBeInTheDocument();
    expect(screen.getByRole('status')).toBeInTheDocument(); // Loading spinner
  });

  test('renders recommendations when data is loaded', async () => {
    render(
      <RecommendationCarousel
        apiUrl="https://api.example.com"
        userId="test-user"
      />
    );
    
    // Wait for recommendations to load
    await waitFor(() => {
      expect(screen.getByText('Wireless Headphones')).toBeInTheDocument();
    });
    
    expect(screen.getByText('Smart Watch')).toBeInTheDocument();
    expect(screen.getByText('$89.99')).toBeInTheDocument();
    expect(screen.getByText('$199.99')).toBeInTheDocument();
  });

  test('renders empty state when no recommendations are available', async () => {
    // Mock fetch to return empty recommendations
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          user_id: 'test-user',
          recommendations: [],
        }),
      })
    ) as jest.Mock;
    
    render(
      <RecommendationCarousel
        apiUrl="https://api.example.com"
        userId="test-user"
      />
    );
    
    // Wait for recommendations to load
    await waitFor(() => {
      expect(screen.getByText('No recommendations available.')).toBeInTheDocument();
    });
  });

  test('renders error state when API request fails', async () => {
    // Mock fetch to return an error
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: false,
        statusText: 'Not Found',
      })
    ) as jest.Mock;
    
    render(
      <RecommendationCarousel
        apiUrl="https://api.example.com"
        userId="test-user"
      />
    );
    
    // Wait for error to show
    await waitFor(() => {
      expect(screen.getByText(/Error fetching recommendations/)).toBeInTheDocument();
    });
  });

  test('calls onItemClick when an item is clicked', async () => {
    const handleItemClick = jest.fn();
    
    render(
      <RecommendationCarousel
        apiUrl="https://api.example.com"
        userId="test-user"
        onItemClick={handleItemClick}
        testRecommendations={mockRecommendations}
      />
    );
    
    // Click on the first item
    userEvent.click(screen.getByText('Wireless Headphones'));
    
    // Check if onItemClick was called with the correct item
    expect(handleItemClick).toHaveBeenCalledWith(mockRecommendations[0]);
  });

  test('uses testRecommendations when provided', () => {
    render(
      <RecommendationCarousel
        apiUrl="https://api.example.com"
        userId="test-user"
        testRecommendations={mockRecommendations}
      />
    );
    
    // No loading state, recommendations shown immediately
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByText('Wireless Headphones')).toBeInTheDocument();
    expect(screen.getByText('Smart Watch')).toBeInTheDocument();
    
    // Fetch should not be called
    expect(global.fetch).not.toHaveBeenCalled();
  });
});