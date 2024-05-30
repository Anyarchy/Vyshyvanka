(function (window, document, undefined) {
    if (L.Canvas) {
        L.Canvas.include({
            _fillStroke: function (ctx, layer) {

                var options = layer.options

                if (options.fill) {
                    ctx.globalAlpha = options.fillOpacity
                    ctx.fillStyle = options.fillColor || options.color

                    ctx.fill(options.fillRule || 'evenodd')
                }

                if (options.stroke && options.weight !== 0) {
                    if (ctx.setLineDash) {
                        ctx.setLineDash(layer.options && layer.options._dashArray || [])
                    }

                    ctx.globalAlpha = options.opacity
                    ctx.lineWidth = options.weight
                    ctx.strokeStyle = options.color
                    ctx.lineCap = options.lineCap
                    ctx.lineJoin = options.lineJoin
                    ctx.stroke()
                    if (options.imgId) {
                        var img = document.getElementById(options.imgId);
                        var bounds = layer._rawPxBounds;
                        var size = bounds.getSize();
                        var imgWidth = img.width;
                        var imgHeight = img.height;

                        // Calculate scale factors
                        var scaleX = size.x / imgWidth;
                        var scaleY = size.y / imgHeight;

                        // Choose the smaller scale to ensure the image fits within the figure
                        var scale = Math.max(scaleX, scaleY);

                        // Calculate the dimensions of the scaled image
                        var scaledWidth = imgWidth * scale;
                        var scaledHeight = imgHeight * scale;

                        // Calculate the position to center the image
                        var offsetX = bounds.min.x + (size.x - scaledWidth) / 2;
                        var offsetY = bounds.min.y + (size.y - scaledHeight) / 2;

                        ctx.save(); // so we can remove the clipping
                        ctx.clip();
                        ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

                        // ctx.fillRect(bounds.min.x, bounds.min.y, size.x, scaledHeight)
                        ctx.restore()
                    }
                }
            }
        })
    }
}(this, document))

