// Mock file for lucide-react icons
const createIconMock = (name) => {
  const Icon = ({ size = 24, className, ...props }) => {
    return {
      $$typeof: Symbol.for('react.element'),
      type: 'svg',
      key: null,
      ref: null,
      props: {
        ...props,
        size,
        className,
        'data-testid': `${name}-icon`,
        children: `${name} Icon`,
      },
      _owner: null,
    };
  };
  Icon.displayName = `${name}Icon`;
  return Icon;
};

// Create mocks for icons used in the components
const ChevronLeft = createIconMock('ChevronLeft');
const ChevronRight = createIconMock('ChevronRight');
const ImageIcon = createIconMock('Image');

module.exports = {
  ChevronLeft,
  ChevronRight,
  ImageIcon,
};