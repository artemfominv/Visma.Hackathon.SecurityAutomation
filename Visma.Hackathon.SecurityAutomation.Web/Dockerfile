FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
USER $APP_UID
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["Visma.Hackathon.SecurityAutomation.Web/Visma.Hackathon.SecurityAutomation.Web.csproj", "Visma.Hackathon.SecurityAutomation.Web/"]
RUN dotnet restore "Visma.Hackathon.SecurityAutomation.Web/Visma.Hackathon.SecurityAutomation.Web.csproj"
COPY . .
WORKDIR "/src/Visma.Hackathon.SecurityAutomation.Web"
RUN dotnet build "Visma.Hackathon.SecurityAutomation.Web.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "Visma.Hackathon.SecurityAutomation.Web.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Visma.Hackathon.SecurityAutomation.Web.dll"]
