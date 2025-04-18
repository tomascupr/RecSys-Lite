# RecSys-Lite Widget

A React component library for displaying recommendations from RecSys-Lite API in your e-commerce applications.

## Features

- 🔄 Responsive carousel for displaying product recommendations
- 🎨 Highly customizable with CSS classes
- 🧩 Easy integration with React applications 
- 💻 Compatible with any RecSys-Lite API endpoint
- 📏 Lightweight with minimal dependencies

## Installation

```bash
npm install recsys-lite-widget
```

## Usage

```jsx
import { RecommendationCarousel } from 'recsys-lite-widget';

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

## Customization

The component is built with flexibility in mind. You can customize its appearance using the provided className props:

```jsx
<RecommendationCarousel 
  apiUrl="https://your-recsys-lite-api.com"
  userId="user123"
  className="bg-gray-100 p-4 rounded-lg"
  containerClassName="gap-4"
  cardClassName="border-blue-500 hover:shadow-lg"
/>
```

## Fetching Additional Item Details

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

## Development

```bash
# Install dependencies
npm install

# Start Storybook for development
npm run storybook

# Build for production
npm run build
```

## License

MIT