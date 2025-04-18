// Mock for embla-carousel-react
const mockScrollPrev = jest.fn();
const mockScrollNext = jest.fn();

const useEmblaCarousel = jest.fn().mockImplementation(() => {
  return [
    jest.fn(), // ref function
    { 
      scrollPrev: mockScrollPrev,
      scrollNext: mockScrollNext,
      canScrollPrev: true,
      canScrollNext: true,
    }
  ];
});

useEmblaCarousel.mockScrollPrev = mockScrollPrev;
useEmblaCarousel.mockScrollNext = mockScrollNext;

module.exports = useEmblaCarousel;
module.exports.default = useEmblaCarousel;