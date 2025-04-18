# RecSys-Lite Widget

A React component library for displaying recommendations from RecSys-Lite API in your e-commerce applications.

![npm](https://img.shields.io/npm/v/recsys-lite-widget)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Storybook](https://img.shields.io/badge/storybook-deployed-ff4785)
![Coverage](https://img.shields.io/badge/coverage-70%25-yellow)

## Features

- 🔄 Responsive carousel for displaying product recommendations
- 🎨 Beautifully styled with shadcn/ui components
- 🧩 Easy integration with React applications 
- 💻 Compatible with any RecSys-Lite API endpoint
- 📏 Lightweight with minimal dependencies
- ♿ Fully accessible with ARIA attributes
- 🌙 Dark mode support
- ✅ Comprehensive test coverage

## Installation

```bash
npm install recsys-lite-widget
```

## Usage

```jsx
import { RecommendationCarousel } from 'recsys-lite-widget';

// Import styles (required)
import 'recsys-lite-widget/dist/style.css';

function App() {
  return (
    <div className="container mx-auto p-4">
      <RecommendationCarousel 
        apiUrl="https://your-recsys-lite-api.com"
        userId="user123"
        count={5}
        title="Recommended For You"
        onItemClick={(item) => console.log('Clicked item:', item)}
      />
    </div>
  );
}
```

## Props

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `apiUrl` | string | required | Base URL of your RecSys-Lite API |
| `userId` | string | required | User ID to get recommendations for |
| `count` | number | 10 | Number of recommendations to display |
| `title` | string | 'Recommended For You' | Title for the recommendations section |
| `className` | string | '' | Additional CSS class for the component |
| `containerClassName` | string | '' | Additional CSS class for the carousel container |
| `cardClassName` | string | '' | Additional CSS class for each recommendation card |
| `onItemClick` | function | undefined | Callback when a recommendation is clicked |
| `fetchItemDetails` | function | undefined | Custom function to fetch additional item details |
| `testRecommendations` | array | undefined | Mock data for testing/development (bypasses API call) |

## Item Structure

The component expects recommendations in the following format:

```typescript
interface Recommendation {
  item_id: string;     // Required: Unique ID for the item
  score: number;       // Required: Recommendation score/confidence
  title?: string;      // Optional: Item title
  image_url?: string;  // Optional: URL to item image
  price?: number;      // Optional: Item price
  category?: string;   // Optional: Item category
  brand?: string;      // Optional: Item brand
}
```

## Customization

The component is built with shadcn/ui and Tailwind CSS, giving you a high degree of customization through CSS classes:

```jsx
<RecommendationCarousel 
  apiUrl="https://your-recsys-lite-api.com"
  userId="user123"
  className="bg-gray-100 p-4 rounded-lg"
  containerClassName="gap-4"
  cardClassName="border-blue-500 hover:shadow-lg"
/>
```

You can also use our exposed UI components directly:

```jsx
import { Card, CardContent, CardHeader, Button } from 'recsys-lite-widget';

function CustomCard() {
  return (
    <Card>
      <CardHeader>Custom Card</CardHeader>
      <CardContent>
        <p>Your custom content</p>
        <Button variant="outline">Click me</Button>
      </CardContent>
    </Card>
  );
}
```

## Advanced Usage

### Fetching Additional Item Details

You can provide a custom function to fetch additional details for the recommended items:

```jsx
<RecommendationCarousel 
  apiUrl="https://your-recsys-lite-api.com"
  userId="user123"
  fetchItemDetails={async (itemIds) => {
    const response = await fetch(`https://your-api.com/items?ids=${itemIds.join(',')}`);
    const data = await response.json();
    return data.items.reduce((acc, item) => {
      acc[item.id] = item;
      return acc;
    }, {});
  }}
/>
```

### Testing and Development

The component accepts `testRecommendations` prop which allows you to bypass API calls and use mock data:

```jsx
<RecommendationCarousel 
  apiUrl="https://your-recsys-lite-api.com"
  userId="user123"
  testRecommendations={[
    {
      item_id: 'item1',
      score: 0.95,
      title: 'Wireless Headphones',
      price: 89.99
    },
    // More items...
  ]}
/>
```

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run Storybook
npm run storybook

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Build for production
npm run build
```

## License

Apache License 2.0